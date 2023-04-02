#%%

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim

import numpy as np
from math import log
from itertools import accumulate
import enlighten
from copy import deepcopy

from utils import default_args, dkl
from maze import T_Maze, action_size
from buffer import RecurrentReplayBuffer
from models import Forward, Actor, Critic



class Agent:
    
    def __init__(self, args = default_args):
        
        self.args = args
        self.episodes = 0 ; self.epochs = 0 ; self.steps = 0
        
        self.target_entropy = self.args.target_entropy # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_opt = optim.Adam(params=[self.log_alpha], lr=self.args.alpha_lr, weight_decay=0) 
        
        self.eta = 1
        self.log_eta = torch.tensor([0.0], requires_grad=True)
        
        self.forward = Forward(self.args)
        self.forward_opt = optim.Adam(self.forward.parameters(), lr=self.args.forward_lr, weight_decay=0)   
                           
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
        
        self.train()
        
        self.memory = RecurrentReplayBuffer(self.args)
        self.plot_dict = {
            "args" : self.args,
            "arg_title" : self.args.arg_title,
            "arg_name" : self.args.arg_name,
            "rewards" : [], "spot_names" : [], 
            "accuracy" : [], "complexity" : [], 
            "alpha" : [], "actor" : [], 
            "critic_1" : [], "critic_2" : [], 
            "extrinsic" : [], "intrinsic_curiosity" : [], 
            "intrinsic_entropy" : [], 
            "naive_1" : [], "naive_2" : [], "naive_3" : [], "free" : []}
        
        
        
    def training(self, i):
        manager = enlighten.Manager(width = 150)
        E = manager.counter(total = self.args.epochs, desc = "{} ({}):".format(self.plot_dict["arg_name"], i), unit = "ticks", color = "blue")
        while(True):
            E.update() ; self.episode()
            if(self.epochs >= self.args.epochs): break
        self.plot_dict["rewards"] = list(accumulate(self.plot_dict["rewards"]))
        
        min_max_dict = {key : [] for key in self.plot_dict.keys()}
        for key in min_max_dict.keys():
            if(not key in ["args", "arg_title", "arg_name", "spot_names"]):
                minimum = None ; maximum = None 
                l = self.plot_dict[key]
                l = deepcopy(l)
                l = [_ for _ in l if _ != None]
                if(l != []):
                    if(  minimum == None):  minimum = min(l)
                    elif(minimum > min(l)): minimum = min(l)
                    if(  maximum == None):  maximum = max(l) 
                    elif(maximum < max(l)): maximum = max(l)
                min_max_dict[key] = (minimum, maximum)
        return(self.plot_dict, min_max_dict)
    
    
    
    def episode(self, push = True, verbose = False):
        done = False ; h = None ; prev_a = torch.zeros((1, 1, action_size))
        t_maze = T_Maze(self.args)
        if(verbose): print("\n\n\n\n\nSTART!\n")
        if(verbose): print(t_maze)
        
        for step in range(self.args.max_steps):
            self.steps += 1
            if(not done):
                with torch.no_grad():
                    o = t_maze.obs().unsqueeze(0)
                    a, _, h = self.actor(o, prev_a, h)
                    action = torch.flatten(a).tolist()
                    r, spot_name, done = t_maze.action(action[0], action[1], verbose)
                    no = t_maze.obs().unsqueeze(0)
                    if(push): self.memory.push(o, a, r, no, done, done)
                    prev_a = a
                
            if(self.steps % self.args.steps_per_epoch == 0 and self.episodes != 0):
                #print("episodes: {}. epochs: {}. steps: {}.".format(self.episodes, self.epochs, self.steps))
                plot_data = self.epoch(batch_size = self.args.batch_size)
                if(plot_data == False): print("Not getting an epoch!")
                else:
                    l, e, ic, ie, naive_1, naive_2, naive_3, free = plot_data
                    if(self.epochs == 1 or self.epochs >= self.args.epochs or self.epochs % self.args.keep_data == 0):
                        self.plot_dict["rewards"].append(r)
                        self.plot_dict["spot_names"].append(spot_name)
                        self.plot_dict["accuracy"].append(l[0][0])
                        self.plot_dict["complexity"].append(l[0][1])
                        self.plot_dict["alpha"].append(l[0][2])
                        self.plot_dict["actor"].append(l[0][3])
                        self.plot_dict["critic_1"].append(l[0][4])
                        self.plot_dict["critic_2"].append(l[0][5])
                        self.plot_dict["extrinsic"].append(e)
                        self.plot_dict["intrinsic_curiosity"].append(ic)
                        self.plot_dict["intrinsic_entropy"].append(ie)
                        self.plot_dict["naive_1"].append(naive_1)
                        self.plot_dict["naive_2"].append(naive_2)
                        self.plot_dict["naive_3"].append(naive_3)
                        self.plot_dict["free"].append(free)    
        self.episodes += 1
    
    
    
    def epoch(self, batch_size):
                        
        batch = self.memory.sample(batch_size)
        if(batch == False): return(False)
        
        self.epochs += 1

        obs, actions, rewards, dones, masks = batch
        
        #print("\n\n")
        #print(obs.shape, actions.shape, rewards.shape, dones.shape, masks.shape)
        #print("\n\n")
        
        next_obs = obs[:,1:]
        obs = obs[:,:-1]
        
        prev_actions = torch.cat([torch.zeros(actions[:,0].unsqueeze(1).shape), actions[:,:-1]], dim = 1)
        
        
                            
        # Train forward
        pred_obs, mu_b, std_b, log_prob_func = self.forward(obs, prev_actions, actions)   
        if(self.args.accuracy == "mse"):      accuracy = F.mse_loss(pred_obs, next_obs, reduction = "none").sum(-1).unsqueeze(-1)
        if(self.args.accuracy == "log_prob"): accuracy = -log_prob_func(next_obs)
        complexity = dkl(mu_b, std_b, torch.zeros(mu_b.shape),  self.args.sigma * torch.ones(std_b.shape))
                
        accuracy = accuracy * masks
        accuracy_loss = accuracy.mean()
        complexity = complexity * masks
        complexity_loss = self.args.beta * complexity.mean()
        print(self.args.beta)
        forward_loss = accuracy_loss + complexity_loss
        if(self.args.beta == 0): complexity = None ; complexity_loss = None
        
        self.forward_opt.zero_grad()
        forward_loss.backward()
        self.forward_opt.step()
        
                        
        
        # Get curiosity  
        naive_1_curiosity = self.args.naive_1_eta * accuracy
        
        _, mu_a, std_a, _ = self.forward(obs, prev_actions, actions)    
        naive_2_curiosity = self.args.naive_2_eta * torch.pow(mu_a - mu_b, 2).sum(-1).unsqueeze(-1)
        naive_3_curiosity = self.args.naive_2_eta * (.5 * (torch.pow(mu_a - mu_b, 2).sum(-1).unsqueeze(-1) - 1))
        
        dkl_changes = dkl(mu_a, std_a, mu_b, std_b).sum(-1).unsqueeze(-1)
        free_curiosity = self.args.free_eta * dkl_changes   
        
        if(self.args.curiosity == "naive_1"):   curiosity = naive_1_curiosity
        elif(self.args.curiosity == "naive_2"): curiosity = naive_2_curiosity
        elif(self.args.curiosity == "naive_3"): curiosity = naive_3_curiosity
        elif(self.args.curiosity == "free"):    curiosity = free_curiosity
        else:                                   curiosity = torch.zeros(rewards.shape)
        curiosity *= masks
        
        extrinsic = torch.mean(rewards).item()
        intrinsic_curiosity = curiosity.mean().item()
        rewards += curiosity
        
        
                
        # Train critics
        with torch.no_grad():
            next_action, log_pis_next, _ = self.actor(next_obs, actions)
            Q_target1_next = self.critic1_target(next_obs, actions, next_action)
            Q_target2_next = self.critic2_target(next_obs, actions, next_action)
            Q_target_next = torch.min(Q_target1_next, Q_target2_next)
            if self.args.alpha == None: Q_targets = rewards + (self.args.GAMMA * (1 - dones) * (Q_target_next - self.alpha * log_pis_next))
            else:                       Q_targets = rewards + (self.args.GAMMA * (1 - dones) * (Q_target_next - self.args.alpha * log_pis_next))
        
        Q_1 = self.critic1(obs, prev_actions, actions)
        critic1_loss = 0.5*F.mse_loss(Q_1*masks, Q_targets*masks)
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()
        
        Q_2 = self.critic2(obs, prev_actions, actions)
        critic2_loss = 0.5*F.mse_loss(Q_2*masks, Q_targets*masks)
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()
        
        
        
        # Train alpha
        if self.args.alpha == None:
            actions_pred, log_pis, _ = self.actor(obs, prev_actions)
            alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy))*masks
            alpha_loss = alpha_loss.mean() / masks.mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = torch.exp(self.log_alpha) 
        else:
            alpha_loss = None
            
            
        
        # Train actor
        if self.epochs % self.args.d == 0:
            if self.args.alpha == None: alpha = self.alpha 
            else:                       
                alpha = self.args.alpha
            actions_pred, log_pis, _ = self.actor(obs, prev_actions)

            if self.args.action_prior == "normal":
                loc = torch.zeros(action_size, dtype=torch.float64)
                scale_tril = torch.tensor([[1, 0], [1, 1]], dtype=torch.float64)
                policy_prior = MultivariateNormal(loc=loc, scale_tril=scale_tril)
                policy_prior_log_probs = policy_prior.log_prob(actions_pred).unsqueeze(-1)
            elif self.args.action_prior == "uniform":
                policy_prior_log_probs = 0.0
            Q = torch.min(
                self.critic1(obs, prev_actions, actions_pred), 
                self.critic2(obs, prev_actions, actions_pred)).mean(-1).unsqueeze(-1)
            intrinsic_entropy = torch.mean((alpha * log_pis)*masks).item()
            actor_loss = (alpha * log_pis - policy_prior_log_probs - Q)*masks
            actor_loss = actor_loss.mean() / masks.mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.soft_update(self.critic1, self.critic1_target, self.args.tau)
            self.soft_update(self.critic2, self.critic2_target, self.args.tau)
            
        else:
            intrinsic_entropy = None
            actor_loss = None
        
        if(accuracy_loss != None): 
            accuracy_loss = accuracy_loss.item()
        if(complexity_loss != None): 
            complexity_loss = complexity_loss.item()
            if(self.args.beta == 0): complexity_loss = None
        if(alpha_loss != None): alpha_loss = alpha_loss.item()
        if(actor_loss != None): actor_loss = actor_loss.item()
        if(critic1_loss != None): 
            critic1_loss = critic1_loss.item()
            critic1_loss = log(critic1_loss) if critic1_loss > 0 else critic1_loss
        if(critic2_loss != None): 
            critic2_loss = critic2_loss.item()
            critic2_loss = log(critic2_loss) if critic2_loss > 0 else critic2_loss
        losses = np.array([[accuracy_loss, complexity_loss, alpha_loss, actor_loss, critic1_loss, critic2_loss]])
        
        naive_1_curiosity = naive_1_curiosity.mean().item()
        naive_2_curiosity = naive_2_curiosity.mean().item()
        naive_3_curiosity = naive_3_curiosity.mean().item()
        free_curiosity = free_curiosity.mean().item()
        if(free_curiosity == 0): free_curiosity = None
        
        return(losses, extrinsic, intrinsic_curiosity, intrinsic_entropy, naive_1_curiosity, naive_2_curiosity, naive_3_curiosity, free_curiosity)
                     
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
        
        
        
if __name__ == "__main__":
    agent = Agent()
# %%
