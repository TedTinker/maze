#%%

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal
import torch.optim as optim
from torch.distributions import Normal

import numpy as np
from math import log
from itertools import accumulate
from copy import deepcopy
from math import exp

from utils import args, default_args, dkl, print
from easy_maze import Easy_Maze, action_size
from easy_buffer import RecurrentReplayBuffer
from easy_models import Forward, Actor, Actor_HQ, Critic, Critic_HQ



class Agent:
    
    def __init__(self, i, args = default_args):
        
        self.agent_num = i
        self.args = args
        self.episodes = 0 ; self.epochs = 0 ; self.steps = 0
        self.maze_name = self.args.maze_list[0]
        self.maze = Easy_Maze(self.maze_name, args = args)
        
        self.target_entropy = args.target_entropy # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_opt = optim.Adam(params=[self.log_alpha], lr=args.alpha_lr, weight_decay=0) 
        
        self.eta = 1
        self.log_eta = torch.tensor([0.0], requires_grad=True)
        
        self.forward = Forward(args)
        self.forward_opt = optim.Adam(self.forward.parameters(), lr=args.forward_lr, weight_decay=0)  
                           
        self.actor = Actor_HQ(args) if args.actor_hq else Actor(args)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=args.actor_lr, weight_decay=0)     
        
        self.critic1 = Critic_HQ(args) if args.critic_hq else Critic(args)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=args.critic_lr, weight_decay=0)
        self.critic1_target = Critic_HQ(args) if args.critic_hq else Critic(args)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic_HQ(args) if args.critic_hq else Critic(args)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=args.critic_lr, weight_decay=0) 
        self.critic2_target = Critic_HQ(args) if args.critic_hq else Critic(args)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.train()
        
        self.memory = RecurrentReplayBuffer(args)
        self.plot_dict = {
            "args" : args,
            "arg_title" : args.arg_title,
            "arg_name" : args.arg_name,
            "pred_lists" : {}, "pos_lists" : {}, 
            "agent_lists" : {"forward" : Forward, "actor" : Actor_HQ if args.actor_hq else Actor, "critic" : Critic_HQ if args.critic_hq else Critic},
            "rewards" : [], "spot_names" : [], 
            "accuracy" : [], "complexity" : [],
            "alpha" : [], "actor" : [], 
            "critic_1" : [], "critic_2" : [], 
            "extrinsic" : [], "intrinsic_curiosity" : [], 
            "intrinsic_entropy" : [], 
            "naive" : [], "free" : []}
        
        
        
    def training(self, q):
        
        self.pred_episodes_hq() if self.args.actor_hq else self.pred_episodes()
        self.pos_episodes_hq() if self.args.actor_hq else self.pos_episodes()
        self.save_agent()
        while(True):
            cumulative_epochs = 0
            prev_maze_name = self.maze_name
            for j, epochs in enumerate(self.args.epochs): 
                cumulative_epochs += epochs
                if(self.epochs < cumulative_epochs): self.maze_name = self.args.maze_list[j] ; break
            if(prev_maze_name != self.maze_name): 
                self.pred_episodes_hq() if self.args.actor_hq else self.pred_episodes()
                self.pos_episodes_hq() if self.args.actor_hq else self.pos_episodes()
                self.maze.maze.stop()
                self.maze = Easy_Maze(self.maze_name, args = self.args)
                self.pred_episodes_hq() if self.args.actor_hq else self.pred_episodes()
                self.pos_episodes_hq() if self.args.actor_hq else self.pos_episodes()
            self.training_episode()
            percent_done = str(self.epochs / sum(self.args.epochs))
            q.put((self.agent_num, percent_done))
            if(self.epochs >= sum(self.args.epochs)): break
            if(self.epochs % self.args.epochs_per_pred_list == 0): self.pred_episodes_hq() if self.args.actor_hq else self.pred_episodes()
            if(self.epochs % self.args.epochs_per_pos_list == 0): self.pos_episodes_hq() if self.args.actor_hq else self.pos_episodes()
            if(self.epochs % self.args.epochs_per_agent_list == 0): self.save_agent()
        self.plot_dict["rewards"] = list(accumulate(self.plot_dict["rewards"]))
        self.pred_episodes_hq() if self.args.actor_hq else self.pred_episodes()
        self.pos_episodes_hq() if self.args.actor_hq else self.pos_episodes()
        self.save_agent()
                
        self.min_max_dict = {key : [] for key in self.plot_dict.keys()}
        for key in self.min_max_dict.keys():
            if(not key in ["args", "arg_title", "arg_name", "pred_lists", "pos_lists", "agent_lists", "spot_names"]):
                minimum = None ; maximum = None 
                l = self.plot_dict[key]
                l = deepcopy(l)
                l = [_ for _ in l if _ != None]
                if(l != []):
                    if(  minimum == None):  minimum = min(l)
                    elif(minimum > min(l)): minimum = min(l)
                    if(  maximum == None):  maximum = max(l) 
                    elif(maximum < max(l)): maximum = max(l)
                self.min_max_dict[key] = (minimum, maximum)
                
                
                
    def save_agent(self):
        if(self.args.agents_per_agent_list != -1 and self.agent_num > self.args.agents_per_agent_list): return
        self.plot_dict["agent_lists"]["{}_{}".format(self.agent_num, self.epochs)] = self.state_dict
                       
       
                
    def step_in_episode(self, prev_a, h_actor, push, verbose):
        with torch.no_grad():
            o = self.maze.obs().unsqueeze(0)
            a, _, h_actor = self.actor(o, prev_a, h_actor)
            action = torch.flatten(a).tolist()
            r, spot_name, done, action_name = self.maze.action(action[0], action[1], verbose)
            no = self.maze.obs().unsqueeze(0)
            if(push): self.memory.push(o, a, r, no, done, done)
        return(a, h_actor, r, spot_name, done, action_name)
    
    
    
    def step_in_episode_hq(self, prev_a, h_q_m1, push, verbose):
        with torch.no_grad():
            o = self.maze.obs().unsqueeze(0)
            _, _, h_q = self.forward(o, prev_a, h_q_m1)
            a, _, _ = self.actor(h_q)
            action = torch.flatten(a).tolist()
            r, spot_name, done, action_name = self.maze.action(action[0], action[1], verbose)
            no = self.maze.obs().unsqueeze(0)
            if(push): self.memory.push(o, a, r, no, done, done)
        return(a, h_q, r, spot_name, done, action_name)
    
    
    
    def pred_episodes(self):
        with torch.no_grad():
            if(self.args.agents_per_pred_list != -1 and self.agent_num > self.args.agents_per_pred_list): return
            pred_lists = []
            for episode in range(self.args.episodes_in_pred_list):
                done = False ; prev_a = torch.zeros((1, 1, action_size))
                h_actor = torch.zeros((1, 1, self.args.hidden_size))
                h_q     = torch.zeros((1, 1, self.args.hidden_size))
                self.maze.begin()
                pred_list = [(None, self.maze.obs(), None, None, None, None)]
                for step in range(self.args.max_steps):
                    if(not done): 
                        o = self.maze.obs()
                        a, h_actor, _, _, done, action_name = self.step_in_episode(prev_a, h_actor, push = False, verbose = False)
                        (zp_mu, zp_std), (zq_mu, zq_std), h_q_p1 = self.forward(o, prev_a, h_q)
                        zp_mu_pred, zp_preds = self.forward.get_preds(a, zp_mu, zp_std, h_q, quantity = self.args.samples_per_pred)
                        zq_mu_pred, zq_preds = self.forward.get_preds(a, zq_mu, zq_std, h_q, quantity = self.args.samples_per_pred)
                        pred_list.append((action_name, self.maze.obs(), zp_mu_pred, zp_preds, zq_mu_pred, zq_preds))
                        prev_a = a ; h_q = h_q_p1
                pred_lists.append(pred_list)
            self.plot_dict["pred_lists"]["{}_{}_{}".format(self.agent_num, self.epochs, self.maze.name)] = pred_lists
            
            
            
    def pred_episodes_hq(self):
        with torch.no_grad():
            if(self.args.agents_per_pred_list != -1 and self.agent_num > self.args.agents_per_pred_list): return
            pred_lists = []
            for episode in range(self.args.episodes_in_pred_list):
                done = False ; prev_a = torch.zeros((1, 1, action_size))
                h_q     = torch.zeros((1, 1, self.args.hidden_size))
                self.maze.begin()
                pred_list = [(None, self.maze.obs(), None, None, None, None)]
                for step in range(self.args.max_steps):
                    if(not done): 
                        o = self.maze.obs()
                        a, h_q_p1, _, _, done, action_name = self.step_in_episode_hq(prev_a, h_q, push = False, verbose = False)
                        (zp_mu, zp_std), (zq_mu, zq_std), h_q_p1 = self.forward(o, prev_a, h_q)
                        zp_mu_pred, zp_preds = self.forward.get_preds(a, zp_mu, zp_std, h_q, quantity = self.args.samples_per_pred)
                        zq_mu_pred, zq_preds = self.forward.get_preds(a, zq_mu, zq_std, h_q, quantity = self.args.samples_per_pred)
                        pred_list.append((action_name, self.maze.obs(), zp_mu_pred, zp_preds, zq_mu_pred, zq_preds))
                        prev_a = a ; h_q = h_q_p1
                pred_lists.append(pred_list)
            self.plot_dict["pred_lists"]["{}_{}_{}".format(self.agent_num, self.epochs, self.maze.name)] = pred_lists
    
    
    
    def pos_episodes(self):
        if(self.args.agents_per_pos_list != -1 and self.agent_num > self.args.agents_per_pos_list): return
        pos_lists = []
        for episode in range(self.args.episodes_in_pos_list):
            done = False ; prev_a = torch.zeros((1, 1, action_size))
            h_actor = torch.zeros((1, 1, self.args.hidden_size))
            self.maze.begin()
            pos_list = [self.maze.agent_pos]
            for step in range(self.args.max_steps):
                if(not done): prev_a, h_actor, _, _, done, _ = self.step_in_episode(prev_a, h_actor, push = False, verbose = False)
                pos_list.append(self.maze.agent_pos)
            pos_lists.append(pos_list)
        self.plot_dict["pos_lists"]["{}_{}_{}".format(self.agent_num, self.epochs, self.maze.name)] = pos_lists
        
        
        
    def pos_episodes_hq(self):
        if(self.args.agents_per_pos_list != -1 and self.agent_num > self.args.agents_per_pos_list): return
        pos_lists = []
        for episode in range(self.args.episodes_in_pos_list):
            done = False ; prev_a = torch.zeros((1, 1, action_size))
            h_q = torch.zeros((1, 1, self.args.hidden_size))
            self.maze.begin()
            pos_list = [self.maze.agent_pos]
            for step in range(self.args.max_steps):
                if(not done): prev_a, h_q, _, _, done, _ = self.step_in_episode_hq(prev_a, h_q, push = False, verbose = False)
                pos_list.append(self.maze.agent_pos)
            pos_lists.append(pos_list)
        self.plot_dict["pos_lists"]["{}_{}_{}".format(self.agent_num, self.epochs, self.maze.name)] = pos_lists
    
    
    
    def training_episode(self, push = True, verbose = False):
        done = False ; prev_a = torch.zeros((1, 1, action_size)) ; cumulative_r = 0
        h = torch.zeros((1, 1, self.args.hidden_size))
        self.maze.begin()
        if(verbose): print("\n\n\n\n\nSTART!\n")
        if(verbose): print(self.maze)
        
        for step in range(self.args.max_steps):
            self.steps += 1
            if(not done):
                if(self.args.actor_hq): prev_a, h, r, spot_name, done, _ = self.step_in_episode_hq(prev_a, h, push, verbose)
                else:                   prev_a, h, r, spot_name, done, _ = self.step_in_episode(   prev_a, h, push, verbose)
                cumulative_r += r
                
            if(self.steps % self.args.steps_per_epoch == 0 and self.episodes != 0):
                #print("episodes: {}. epochs: {}. steps: {}.".format(self.episodes, self.epochs, self.steps))
                plot_data = self.epoch(batch_size = self.args.batch_size)
                if(plot_data == False): print("Not getting an epoch!")
                else:
                    l, e, ic, ie, naive, free = plot_data
                    if(self.epochs == 1 or self.epochs >= sum(self.args.epochs) or self.epochs % self.args.keep_data == 0):
                        self.plot_dict["accuracy"].append(l[0][0])
                        self.plot_dict["complexity"].append(l[0][1])
                        self.plot_dict["alpha"].append(l[0][2])
                        self.plot_dict["actor"].append(l[0][3])
                        self.plot_dict["critic_1"].append(l[0][4])
                        self.plot_dict["critic_2"].append(l[0][5])
                        self.plot_dict["extrinsic"].append(e)
                        self.plot_dict["intrinsic_curiosity"].append(ic)
                        self.plot_dict["intrinsic_entropy"].append(ie)
                        self.plot_dict["naive"].append(naive)
                        self.plot_dict["free"].append(free)    
        self.plot_dict["rewards"].append(r)
        self.plot_dict["spot_names"].append(spot_name)
        self.episodes += 1
    
    
    
    def epoch(self, batch_size):
                        
        batch = self.memory.sample(batch_size)
        if(batch == False): return(False)
        
        self.epochs += 1

        all_obs, actions, rewards, dones, masks = batch
        next_obs = all_obs[:,1:]
        obs = all_obs[:,:-1]
        all_actions = torch.cat([torch.zeros(actions[:,0].unsqueeze(1).shape), actions], dim = 1)
        prev_actions = all_actions[:,:-1]
        episodes = rewards.shape[0] ; steps = rewards.shape[1]
        
        #print("\n\n")
        #print("all obs: {}. next obs: {}. obs: {}. all actions: {}. prev actions: {}. actions: {}. rewards: {}. dones: {}. masks: {}.".format(
        #    all_obs.shape, next_obs.shape, obs.shape, all_actions.shape, prev_actions.shape, actions.shape, rewards.shape, dones.shape, masks.shape))
        #print("\n\n")
        
        

        # Train forward
        h_qs = [torch.zeros((episodes, 1, self.args.hidden_size)).to(obs.device)]
        zp_mus = [] ; zp_stds = []
        zq_mus = [] ; zq_stds = [] ; zq_pred_obs = []
        for step in range(steps):
            (zp_mu, zp_std), (zq_mu, zq_std), h_q_p1 = self.forward(obs[:, step], prev_actions[:, step], h_qs[-1])
            _, zq_preds = self.forward.get_preds(actions[:, step], zq_mu, zq_std, h_qs[-1], quantity = self.args.elbo_num)
            zp_mus.append(zp_mu) ; zp_stds.append(zp_std)
            zq_mus.append(zq_mu) ; zq_stds.append(zq_std) ; zq_pred_obs.append(torch.cat(zq_preds, -1))
            h_qs.append(h_q_p1)
        h_qs.append(h_qs.pop(0)) ; h_qs = torch.cat(h_qs, dim = 1) ; next_hqs = h_qs[:, 1:] ; hqs = h_qs[:, :-1]
        zp_mus = torch.cat(zp_mus, dim = 1) ; zp_stds = torch.cat(zp_stds, dim = 1)
        zq_mus = torch.cat(zq_mus, dim = 1) ; zq_stds = torch.cat(zq_stds, dim = 1) ; zq_pred_obs = torch.cat(zq_pred_obs, dim = 1)
                
        next_obs_tiled = torch.tile(next_obs, (1, 1, self.args.elbo_num))
                
        accuracy_for_naive  = F.binary_cross_entropy_with_logits(zq_pred_obs, (next_obs_tiled+1)/2, reduction = "none").mean(-1).unsqueeze(-1) * masks / self.args.elbo_num
        accuracy            = accuracy_for_naive.mean()
        complexity_for_free = dkl(zq_mus, zq_stds, zp_mus, zp_stds).mean(-1).unsqueeze(-1) * masks
        complexity          = self.args.beta * complexity_for_free.mean()        
                        
        self.forward_opt.zero_grad()
        (accuracy + complexity).backward()
        self.forward_opt.step()
        
        if(self.args.beta == 0): complexity = None
            
                        
        
        # Get curiosity                  
        naive_curiosity = self.args.naive_eta * accuracy_for_naive  
        if(self.args.dkl_max != None):
            complexity_for_free = torch.clamp(complexity_for_free, min = 0, max = self.args.dkl_max)
        free_curiosity = self.args.free_eta * complexity_for_free
        if(self.args.curiosity == "naive"):  curiosity = naive_curiosity
        elif(self.args.curiosity == "free"): curiosity = free_curiosity
        else:                                curiosity = torch.zeros(rewards.shape)
        extrinsic = torch.mean(rewards).item()
        intrinsic_curiosity = curiosity.mean().item()
        rewards += curiosity
        
        
                
        # Train critics
        with torch.no_grad():
            next_action, log_pis_next, _ = self.actor(next_hqs) if self.args.actor_hq else self.actor(next_obs, actions)
            Q_target1_next = self.critic1_target(next_hqs, next_action) if self.args.critic_hq else self.critic1_target(next_obs, next_action)
            Q_target2_next = self.critic2_target(next_hqs, next_action) if self.args.critic_hq else self.critic2_target(next_obs, next_action)
            Q_target_next = torch.min(Q_target1_next, Q_target2_next)
            if self.args.alpha == None: Q_targets = rewards + (self.args.GAMMA * (1 - dones) * (Q_target_next - self.alpha * log_pis_next))
            else:                       Q_targets = rewards + (self.args.GAMMA * (1 - dones) * (Q_target_next - self.args.alpha * log_pis_next))
        
        Q_1 = self.critic1(hqs.detach(), actions) if self.args.critic_hq else self.critic1(obs, actions)
        critic1_loss = 0.5*F.mse_loss(Q_1*masks, Q_targets*masks)
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()
        
        Q_2 = self.critic2(hqs.detach(), actions) if self.args.critic_hq else self.critic2(obs, actions)
        critic2_loss = 0.5*F.mse_loss(Q_2*masks, Q_targets*masks)
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()
        
        
        
        # Train alpha
        if self.args.alpha == None:
            _, log_pis, _ = self.actor(hqs.detach()) if self.args.actor_hq else self.actor(obs, prev_actions)
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
            actions_pred, log_pis, _ = self.actor(hqs.detach()) if self.args.actor_hq else self.actor(obs, prev_actions)

            if self.args.action_prior == "normal":
                loc = torch.zeros(action_size, dtype=torch.float64)
                scale_tril = torch.tensor([[1, 0], [1, 1]], dtype=torch.float64)
                policy_prior = MultivariateNormal(loc=loc, scale_tril=scale_tril)
                policy_prior_log_probs = policy_prior.log_prob(actions_pred).unsqueeze(-1)
            elif self.args.action_prior == "uniform":
                policy_prior_log_probs = 0.0
            Q = torch.min(
                self.critic1(hqs.detach(), actions_pred) if self.args.critic_hq else self.critic1(obs, actions_pred), 
                self.critic2(hqs.detach(), actions_pred) if self.args.critic_hq else self.critic2(obs, actions_pred)).mean(-1).unsqueeze(-1)
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
        
        if(accuracy != None):   accuracy = accuracy.item()
        if(complexity != None): complexity = complexity.item()
        if(alpha_loss != None): alpha_loss = alpha_loss.item()
        if(actor_loss != None): actor_loss = actor_loss.item()
        if(critic1_loss != None): critic1_loss = log(critic1_loss.item()) if critic1_loss > 0 else critic1_loss.item()
        if(critic2_loss != None): critic2_loss = log(critic2_loss.item()) if critic2_loss > 0 else critic2_loss.item()
        losses = np.array([[accuracy, complexity, alpha_loss, actor_loss, critic1_loss, critic2_loss]])
        
        naive_curiosity = naive_curiosity.mean().item()
        free_curiosity  = free_curiosity.mean().item()
        if(free_curiosity == 0): free_curiosity = None
        
        return(losses, extrinsic, intrinsic_curiosity, intrinsic_entropy, naive_curiosity, free_curiosity)
    
    
                     
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
