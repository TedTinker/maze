#%%

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim
from blitz.losses import kl_divergence_from_nn as b_kl_loss

import numpy as np
from math import log

from utils import default_args, dkl, weights
from buffer import RecurrentReplayBuffer
from models import Forward, Actor, Critic



class Agent:
    
    def __init__(self, action_prior="normal", args = default_args):
        
        self.args = args
        self.steps = 0
        self.action_size = 2
        
        self.target_entropy = self.args.target_entropy # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_opt = optim.Adam(params=[self.log_alpha], lr=self.args.alpha_lr, weight_decay=0) 
        self._action_prior = action_prior
        
        self.eta = 1
        self.log_eta = torch.tensor([0.0], requires_grad=True)
        
        self.forward = Forward(self.args)
        self.forward_opt = optim.Adam(self.forward.parameters(), lr=self.args.forward_lr, weight_decay=0)   
        
        if(self.args.dkl_change_size):
            clone_lr = self.args.clone_lr
            self.forward_clone = Forward(self.args)
            self.clone_opt = optim.Adam(self.forward_clone.parameters(), lr=clone_lr, weight_decay=0)
                           
        self.actor = Actor(self.args)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr, weight_decay=0)     
        
        self.critic1 = Critic(self.args)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=self.args.critic_lr, weight_decay=0)
        self.critic1_target = Critic(self.args)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(self.args)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=self.args.critic_lr, weight_decay=0) 
        self.critic2_target = Critic(self.args)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.restart_memory()
        
    def restart_memory(self):
        self.memory = RecurrentReplayBuffer(self.args)

    def act(self, pos):
        action = self.actor.get_action(pos).detach()
        return action
    
    def learn(self, batch_size,):
                
        self.steps += 1

        obs, actions, rewards, dones, masks = self.memory.sample(batch_size)
        
        next_obs = obs[:,1:]
        obs = obs[:,:-1]
        
        
                            
        # Train forward
        forward_errors = torch.zeros(rewards.shape)
        dkl_loss = 0
        for _ in range(self.args.sample_elbo):
            pred_obs = self.forward(obs, actions)            
            errors = F.mse_loss(pred_obs, next_obs.detach(), reduction = "none") 
            errors = torch.sum(errors, -1).unsqueeze(-1)
            forward_errors += errors / self.args.sample_elbo
            dkl_loss += self.args.dkl_rate * b_kl_loss(self.forward) / self.args.sample_elbo
        forward_errors *= masks.detach()
        mse_loss = forward_errors.sum()
        forward_loss = mse_loss + dkl_loss
        #print("\nMSE: {}. KL: {}.\n".format(mse_loss.item(), dkl_loss if type(dkl_loss == int) else dkl_loss.item()))
        
        old_state_dict = self.forward.state_dict() # For curiosity
        weights_before = weights(self.forward)
        self.forward_opt.zero_grad()
        forward_loss.sum().backward()
        self.forward_opt.step()
        weights_after = weights(self.forward)
                
        dkl_change = dkl(weights_after[0], weights_after[1], weights_before[0], weights_before[1]) + \
            dkl(weights_after[2], weights_after[3], weights_before[2], weights_before[3])
        dkl_changes = torch.tile(dkl_change, rewards.shape)   
        dkl_changes *= masks
        if(dkl_changes.sum().item() != 0): dkl_change = dkl_changes.sum().item()
        
        
        
        # Do we need this? We could just calculate dkl_changes using each error! 
        if(self.args.dkl_change_size == "step"):
            dkl_changes = torch.zeros(rewards.shape)
            for episode in range(rewards.shape[0]):
                for step in range(rewards.shape[1]):
                    if(masks[episode, step] == 0): dkl_changes[episode, step] = 0 ; break
                    self.forward_clone.load_state_dict(old_state_dict)
                    forward_errors_ = torch.zeros(rewards.shape)
                    dkl_loss_ = 0
                    for _ in range(self.args.sample_elbo):
                        pred_obs_ = self.forward_clone(obs[episode, step], actions[episode, step])            
                        errors_ = F.mse_loss(pred_obs_, next_obs.detach()[episode, step], reduction = "none") 
                        errors_ = torch.sum(errors_, -1).unsqueeze(-1)
                        forward_errors_ += errors_ / self.args.sample_elbo
                        dkl_loss_ += self.args.dkl_rate * b_kl_loss(self.forward_clone) / self.args.sample_elbo
                    forward_errors_ *= masks.detach()
                    mse_loss_ = forward_errors_.sum()
                    forward_loss_ = mse_loss_ + dkl_loss_
                    #print("\nMSE: {}. KL: {}.\n".format(mse_loss.item(), dkl_loss if type(dkl_loss == int) else dkl_loss.item()))
            
                    weights_before = weights(self.forward_clone)
                    self.clone_opt.zero_grad()
                    forward_loss_.sum().backward()
                    self.clone_opt.step()
                    weights_after = weights(self.forward_clone)
                    
                    dkl_change = dkl(weights_after[0], weights_after[1], weights_before[0], weights_before[1]) + \
                        dkl(weights_after[2], weights_after[3], weights_before[2], weights_before[3])
                    dkl_changes[episode, step] = dkl_change
            dkl_changes *= masks 
            if(dkl_changes.sum().item() != 0): dkl_change = dkl_changes.sum().item()
                    
        
        
        # Get curiosity          
        naive_curiosity   = self.args.naive_eta   * forward_errors   
        naive_curiosity *= masks.detach() 
        friston_curiosity = self.args.friston_eta * dkl_changes  
        friston_curiosity *= masks.detach()
        if(self.args.curiosity == "naive"):
            curiosity = naive_curiosity
            #print("\nMSE curiosity: {}, {}.\n".format(curiosity.shape, torch.sum(curiosity)))
        elif(self.args.curiosity == "friston"):
            curiosity = friston_curiosity
            #print("\nFEB curiosity: {}, {}.\n".format(curiosity.shape, torch.sum(curiosity)))
        else:
            curiosity = torch.zeros(rewards.shape)
        
        extrinsic = torch.mean(rewards*masks.detach()).item()
        intrinsic_curiosity = curiosity.sum().item()
        rewards += curiosity
        
        
                
        # Train critics
        next_action, log_pis_next = self.actor.evaluate(next_obs.detach())
        Q_target1_next = self.critic1_target(next_obs.detach(), next_action.detach())
        Q_target2_next = self.critic2_target(next_obs.detach(), next_action.detach())
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)
        if self.args.alpha == None: Q_targets = rewards.cpu() + (self.args.GAMMA * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.cpu()))
        else:                       Q_targets = rewards.cpu() + (self.args.GAMMA * (1 - dones.cpu()) * (Q_target_next.cpu() - self.args.alpha * log_pis_next.cpu()))
        
        Q_1 = self.critic1(obs.detach(), actions.detach()).cpu()
        critic1_loss = 0.5*F.mse_loss(Q_1*masks.detach().cpu(), Q_targets.detach()*masks.detach().cpu())
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()
        
        Q_2 = self.critic2(obs.detach(), actions.detach()).cpu()
        critic2_loss = 0.5*F.mse_loss(Q_2*masks.detach().cpu(), Q_targets.detach()*masks.detach().cpu())
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()
        
        
        
        # Train alpha
        if self.args.alpha == None:
            actions_pred, log_pis = self.actor.evaluate(obs.detach())
            alpha_loss = -(self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu())*masks.detach().cpu()
            alpha_loss = alpha_loss.sum() / masks.sum()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = torch.exp(self.log_alpha) 
        else:
            alpha_loss = None
            
            
        
        # Train actor
        if self.steps % self.args.d == 0:
            if self.args.alpha == None: alpha = self.alpha 
            else:                       
                alpha = self.args.alpha
                actions_pred, log_pis = self.actor.evaluate(obs.detach())

            if self._action_prior == "normal":
                loc = torch.zeros(self.action_size, dtype=torch.float64)
                scale_tril = torch.tensor([[1, 0], [1, 1]], dtype=torch.float64)
                policy_prior = MultivariateNormal(loc=loc, scale_tril=scale_tril)
                policy_prior_log_probs = policy_prior.log_prob(actions_pred.cpu()).unsqueeze(-1)
            elif self._action_prior == "uniform":
                policy_prior_log_probs = 0.0
            Q = torch.min(
                self.critic1(obs.detach(), actions_pred), 
                self.critic2(obs.detach(), actions_pred)).sum(-1).unsqueeze(-1)
            intrinsic_entropy = torch.mean((alpha * log_pis.cpu())*masks.detach().cpu()).item()
            actor_loss = (alpha * log_pis.cpu() - policy_prior_log_probs - Q.cpu())*masks.detach().cpu()
            actor_loss = actor_loss.sum() / masks.sum()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.soft_update(self.critic1, self.critic1_target, self.args.tau)
            self.soft_update(self.critic2, self.critic2_target, self.args.tau)
            
        else:
            intrinsic_entropy = None
            actor_loss = None
        
        if(mse_loss != None): mse_loss = mse_loss.item()
        if(dkl_loss != None): dkl_loss = dkl_loss.item()
        if(alpha_loss != None): alpha_loss = alpha_loss.item()
        if(actor_loss != None): actor_loss = actor_loss.item()
        if(critic1_loss != None): critic1_loss = critic1_loss.item()
        if(critic2_loss != None): critic2_loss = critic2_loss.item()
        losses = np.array([[mse_loss, dkl_loss, alpha_loss, actor_loss, critic1_loss, critic2_loss]])
        
        return(losses, extrinsic, intrinsic_curiosity, intrinsic_entropy, dkl_change, naive_curiosity.sum().detach(), friston_curiosity.sum().detach())
                     
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def state_dict(self):
        return(
            self.forward.state_dict(),
            self.actor.state_dict(),
            self.critic1.state_dict(),
            self.critic1_target.state_dict(),
            self.critic2.state_dict(),
            self.critic2_target.state_dict())

    def load_state_dict(self, state_dict):
        self.forward.load_state_dict(state_dict[0])
        self.actor.load_state_dict(state_dict[1])
        self.critic1.load_state_dict(state_dict[2])
        self.critic1_target.load_state_dict(state_dict[3])
        self.critic2.load_state_dict(state_dict[4])
        self.critic2_target.load_state_dict(state_dict[5])
        self.memory = RecurrentReplayBuffer(self.args)

    def eval(self):
        self.forward.eval()
        self.actor.eval()
        self.critic1.eval()
        self.critic1_target.eval()
        self.critic2.eval()
        self.critic2_target.eval()

    def train(self):
        self.forward.train()
        self.actor.train()
        self.critic1.train()
        self.critic1_target.train()
        self.critic2.train()
        self.critic2_target.train()
        
print("agent.py loaded.")
# %%
