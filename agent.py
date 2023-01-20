#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim
from blitz.losses import kl_divergence_from_nn as b_kl_loss

import numpy as np
from math import log

from utils import args, dkl, weights
from buffer import RecurrentReplayBuffer
from models import Forward, Bayes_Forward, Actor, Critic



class Agent:
    
    def __init__(self, action_prior="normal", args = args):
        
        self.args = args
        self.steps = 0
        self.action_size = 2
        
        self.target_entropy = self.args.target_entropy # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=self.args.alpha_lr, weight_decay=0) 
        self._action_prior = action_prior
        
        self.eta = 1
        self.log_eta = torch.tensor([0.0], requires_grad=True)
        
        if(self.args.bayes):
            self.forward = Bayes_Forward()
        else:
            self.forward = Forward()
        self.forward_optimizer = optim.Adam(self.forward.parameters(), lr=self.args.forward_lr, weight_decay=0)     
                           
        self.actor = Actor()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr, weight_decay=0)     
        
        self.critic1 = Critic()
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.args.critic_lr, weight_decay=0)
        self.critic1_target = Critic()
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic()
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.args.critic_lr, weight_decay=0) 
        self.critic2_target = Critic()
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
        forward_errors *= masks
        mse_loss = forward_errors.sum()
        forward_loss = mse_loss + dkl_loss
        #print("\nMSE: {}. KL: {}.\n".format(mse_loss.item(), dkl_loss if type(dkl_loss == int) else dkl_loss.item()))
        
        weights_before = weights(self.forward)
    
        self.forward_optimizer.zero_grad()
        forward_loss.sum().backward()
        self.forward_optimizer.step()
        
        weights_after = weights(self.forward)
        
        dkl_change = dkl(weights_after[0], weights_after[1], weights_before[0], weights_before[1]) + \
            dkl(weights_after[2], weights_after[3], weights_before[2], weights_before[3])
        dkl_changes = torch.tile(dkl_change, rewards.shape)   
        
        if(dkl_changes.sum().item() != 0):
            dkl_change = log(dkl_changes.sum().item())    
        dkl_changes *= masks 
                        
        if(self.args.naive_curiosity):
            curiosity = self.args.eta * forward_errors
            #print("\nMSE curiosity: {}, {}.\n".format(curiosity.shape, torch.sum(curiosity)))
        else:
            curiosity = self.args.eta * dkl_changes
            #print("\nFEB curiosity: {}, {}.\n".format(curiosity.shape, torch.sum(curiosity)))
                        
        extrinsic = torch.mean(rewards*masks.detach()).item()
        intrinsic_curiosity = torch.mean(curiosity*masks.detach()).item()
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
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        Q_2 = self.critic2(obs.detach(), actions.detach()).cpu()
        critic2_loss = 0.5*F.mse_loss(Q_2*masks.detach().cpu(), Q_targets.detach()*masks.detach().cpu())
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        
        
        # Train alpha
        if self.args.alpha == None:
            actions_pred, log_pis = self.actor.evaluate(obs.detach())
            alpha_loss = -(self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu())*masks.detach().cpu()
            alpha_loss = alpha_loss.sum() / masks.sum()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
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

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.critic1, self.critic1_target, self.args.tau)
            self.soft_update(self.critic2, self.critic2_target, self.args.tau)
            
        else:
            intrinsic_entropy = None
            actor_loss = None
        
        if(mse_loss != None): mse_loss = log(mse_loss.item())
        if(dkl_loss != None): 
            try: dkl_loss = log(dkl_loss.item())
            except: dkl_loss = 0
        if(alpha_loss != None): alpha_loss = alpha_loss.item()
        if(actor_loss != None): actor_loss = actor_loss.item()
        if(critic1_loss != None): critic1_loss = log(critic1_loss.item())
        if(critic2_loss != None): critic2_loss = log(critic2_loss.item())
        losses = np.array([[mse_loss, dkl_loss, alpha_loss, actor_loss, critic1_loss, critic2_loss]])
        
        try:    intrinsic_entropy = (1 if intrinsic_entropy >= 0 else -1) * abs(intrinsic_entropy)**.5
        except: pass
        try:    intrinsic_curiosity = log(intrinsic_curiosity)
        except: pass
        
        return(losses, extrinsic, intrinsic_curiosity, intrinsic_entropy, dkl_change)
                     
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