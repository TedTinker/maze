#%%

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim

import numpy as np
from math import log
from itertools import accumulate
from copy import deepcopy
from math import exp

from utils import default_args, dkl
from hard_maze import Hard_Maze
from hard_buffer import RecurrentReplayBuffer
from hard_models import Forward, Actor, Critic

action_size = 2



class Agent:
    
    def __init__(self, i, args = default_args):
        
        self.agent_num = i
        self.args = args
        self.episodes = 0 ; self.epochs = 0 ; self.steps = 0
        self.maze_name = self.args.maze_list[0]
        self.maze = Hard_Maze(self.maze_name, args = self.args)
        
        self.target_entropy = args.target_entropy # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_opt = optim.Adam(params=[self.log_alpha], lr=args.alpha_lr, weight_decay=0) 
        
        self.eta = 1
        self.log_eta = torch.tensor([0.0], requires_grad=True)
        
        self.forward = Forward(self.args)
        without_zp = [] 
        just_zp = []
        for name, param in self.forward.named_parameters():
            if name.startswith("zp"): just_zp.append(param)
            else:                     without_zp.append(param)
        self.forward_opt = optim.Adam(without_zp, lr=self.args.forward_lr, weight_decay=0)   
        self.zp_opt = optim.Adam(just_zp, lr=self.args.forward_lr, weight_decay=0)   
                           
        self.actor = Actor(args)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=args.actor_lr, weight_decay=0)     
        
        self.critic1 = Critic(args)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=args.critic_lr, weight_decay=0)
        self.critic1_target = Critic(args)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(args)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=args.critic_lr, weight_decay=0) 
        self.critic2_target = Critic(args)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.train()
        
        self.memory = RecurrentReplayBuffer(self.args)
        self.plot_dict = {
            "args" : self.args,
            "arg_title" : self.args.arg_title,
            "arg_name" : self.args.arg_name,
            "pred_lists" : {}, "pos_lists" : {},
            "rewards" : [], "spot_names" : [], 
            "accuracy" : [], "obs_complexity" : [], "zq_complexity" : [], "zp" : [],
            "alpha" : [], "actor" : [], 
            "critic_1" : [], "critic_2" : [], 
            "extrinsic" : [], "intrinsic_curiosity" : [], 
            "intrinsic_entropy" : [], 
            "naive" : [], "free" : []}
        
        
        
    def training(self, q):
        
        self.pred_episodes()
        self.pos_episodes()
        while(True):
            cumulative_epochs = 0
            prev_maze_name = self.maze_name
            for j, epochs in enumerate(self.args.epochs): 
                cumulative_epochs += epochs
                if(self.epochs < cumulative_epochs): self.maze_name = self.args.maze_list[j] ; break
            if(prev_maze_name != self.maze_name): 
                self.pred_episodes()
                self.pos_episodes()
                self.maze.maze.stop()
                self.maze = Hard_Maze(self.maze_name, args = self.args)
                self.pred_episodes()
                self.pos_episodes()
            self.training_episode()
            percent_done = str(self.epochs / sum(self.args.epochs))
            q.put((self.agent_num, percent_done))
            if(self.epochs >= sum(self.args.epochs)): break
            if(self.epochs % self.args.epochs_per_pred_list == 0): self.pred_episodes()
            if(self.epochs % self.args.epochs_per_pos_list == 0): self.pos_episodes()
        self.plot_dict["rewards"] = list(accumulate(self.plot_dict["rewards"]))
        self.pred_episodes()
        self.pos_episodes()
        
        self.min_max_dict = {key : [] for key in self.plot_dict.keys()}
        for key in self.min_max_dict.keys():
            if(not key in ["args", "arg_title", "arg_name", "pred_lists", "pos_lists", "spot_names"]):
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
                
                
                
    def step_in_episode(self, prev_a, h, push, verbose):
        with torch.no_grad():
            o, s = self.maze.obs()
            o = o.unsqueeze(0) ; s = s.unsqueeze(0)
            a, _, h = self.actor(o, s, prev_a, h)
            action = torch.flatten(a).tolist()
            r, spot_name, done = self.maze.action(action[0], action[1], verbose)
            no, ns = self.maze.obs()
            no = no.unsqueeze(0) ; ns = ns.unsqueeze(0)
            if(push): self.memory.push(o, s, a, r, no, ns, done, done)
        return(a, h, r, spot_name, done)
    
    
    
    def pred_episodes(self):
        with torch.no_grad():
            if(self.args.agents_per_pred_list != -1 and self.agent_num >= self.args.agents_per_pred_list): return
            pred_lists = []
            for episode in range(self.args.episodes_in_pred_list):
                done = False ; h = None ; forward_h = None ; prev_a = torch.zeros((1, 1, action_size))
                self.maze.begin()
                pred_list = [(self.maze.obs(), (None, None), (None, None))]
                for step in range(self.args.max_steps):
                    if(not done): 
                        o, s = self.maze.obs()
                        o = o.unsqueeze(0) ; s = s.unsqueeze(0)
                        a, h, _, _, done = self.step_in_episode(prev_a, h, push = False, verbose = False)
                        (_, pred_o_mu, pred_o_std), (_, pred_s_mu, pred_s_std), _, _, forward_h = self.forward(o, s, prev_a, a, forward_h)
                        next_o, next_s = self.maze.obs()
                        pred_list.append(((next_o, next_s), (pred_o_mu, pred_o_std), (pred_s_mu, pred_s_std)))
                        prev_a = a
                pred_lists.append(pred_list)
            self.plot_dict["pred_lists"]["{}_{}".format(self.agent_num, self.epochs)] = pred_lists
    
    
    
    def pos_episodes(self):
        if(self.args.agents_per_pos_list != -1 and self.agent_num >= self.args.agents_per_pos_list): return
        pos_lists = []
        for episode in range(self.args.episodes_in_pos_list):
            done = False ; h = None ; prev_a = torch.zeros((1, 1, action_size))
            self.maze.begin()
            pos_list = [self.maze_name, self.maze.maze.get_pos_yaw_spe()[0]]
            for step in range(self.args.max_steps):
                if(not done): prev_a, h, _, _, done = self.step_in_episode(prev_a, h, push = False, verbose = False)
                pos_list.append(self.maze.maze.get_pos_yaw_spe()[0])
            pos_lists.append(pos_list)
        self.plot_dict["pos_lists"]["{}_{}".format(self.agent_num, self.epochs)] = pos_lists
    
    
    
    def training_episode(self, push = True, verbose = False):
        done = False ; h = None ; prev_a = torch.zeros((1, 1, 2)) ; cumulative_r = 0
        self.maze.begin()
        if(verbose): print("\n\n\n\n\nSTART!\n")
        
        for step in range(self.args.max_steps):
            self.steps += 1
            if(not done):
                prev_a, h, r, spot_name, done = self.step_in_episode(prev_a, h, push, verbose)
                cumulative_r += r
                
            if(self.steps % self.args.steps_per_epoch == 0 and self.episodes != 0):
                #print("episodes: {}. epochs: {}. steps: {}.".format(self.episodes, self.epochs, self.steps))
                plot_data = self.epoch(batch_size = self.args.batch_size)
                if(plot_data == False): print("Not getting an epoch!")
                else:
                    l, e, ic, ie, naive, free = plot_data
                    if(self.epochs == 1 or self.epochs >= sum(self.args.epochs) or self.epochs % self.args.keep_data == 0):
                        self.plot_dict["accuracy"].append(l[0][0])
                        self.plot_dict["obs_complexity"].append(l[0][1])
                        self.plot_dict["zq_complexity"].append(l[0][2])
                        self.plot_dict["zp"].append(l[0][3])
                        self.plot_dict["alpha"].append(l[0][4])
                        self.plot_dict["actor"].append(l[0][5])
                        self.plot_dict["critic_1"].append(l[0][6])
                        self.plot_dict["critic_2"].append(l[0][7])
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

        all_rgbd, all_spe, actions, rewards, dones, masks = batch
        next_rgbd = all_rgbd[:,1:]
        rgbd = all_rgbd[:,:-1]
        next_spe = all_spe[:,1:]
        spe = all_spe[:,:-1]
        all_actions = torch.cat([torch.zeros(actions[:,0].unsqueeze(1).shape), actions], dim = 1)
        prev_actions = all_actions[:,:-1]
        episodes = rewards.shape[0] ; steps = rewards.shape[1]
        
        #print("\n\n")
        #print("all rgbd: {}. next rgbd: {}. rgbd: {}. all spe: {}. next spe: {}. spe: {}. all actions: {}. prev actions: {}. actions: {}. rewards: {}. dones: {}. masks: {}.".format(
        #    all_rgbd.shape, next_rgbd.shape, rgbd.shape, all_spe.shape, next_spe.shape, spe.shape, all_actions.shape, prev_actions.shape, actions.shape, rewards.shape, dones.shape, masks.shape))
        #print("\n\n")
        
        

        # Train forward
        pred_rgbd = [] ; rgbd_mus_b = [] ; rgbd_stds_b = []
        pred_spe = [] ; spe_mus_b = [] ; spe_stds_b = []
        zp_mus = [] ; zp_stds = []
        zq_mus = [] ; zq_stds = [] ; h = None
        for step in range(steps):
            (p_rgbd, rgbd_mu, rgbd_std), (p_spe, spe_mu, spe_std), (zp, zp_mu, zp_std), (_, zq_mu, zq_std), h = self.forward(rgbd[:, step], spe[:, step], prev_actions[:, step], actions[:, step], h)   
            pred_rgbd.append(p_rgbd) ; rgbd_mus_b.append(rgbd_mu) ; rgbd_stds_b.append(rgbd_std)
            pred_spe.append(p_spe) ; spe_mus_b.append(spe_mu) ; spe_stds_b.append(spe_std)
            zp_mus.append(zp_mu) ; zp_stds.append(zp_std)
            zq_mus.append(zq_mu) ; zq_stds.append(zq_std)
        pred_rgbd = torch.cat(pred_rgbd, dim = 1) ; rgbd_mus_b = torch.cat(rgbd_mus_b, dim = 1) ; rgbd_stds_b = torch.cat(rgbd_stds_b, dim = 1)
        pred_spe = torch.cat(pred_spe, dim = 1) ; spe_mus_b = torch.cat(spe_mus_b, dim = 1) ; spe_stds_b = torch.cat(spe_stds_b, dim = 1)
        zp_mus = torch.cat(zp_mus, dim = 1) ; zp_stds = torch.cat(zp_stds, dim = 1)
        zq_mus = torch.cat(zq_mus, dim = 1) ; zq_stds = torch.cat(zq_stds, dim = 1)
        if(self.args.accuracy == "mse"):      
            accuracy  = F.mse_loss(pred_rgbd, next_rgbd, reduction = "none").sum(-1).unsqueeze(-1).flatten(2)
            accuracy += F.mse_loss(pred_spe, next_spe, reduction = "none").sum(-1).unsqueeze(-1)
        if(self.args.accuracy == "log_prob"): 
            accuracy  = 0.5 * (torch.log(2 * np.pi * rgbd_stds_b**2) + ((next_rgbd - rgbd_mus_b) ** 2) / (rgbd_stds_b**2 + 1e-6)).sum(-1).unsqueeze(-1).flatten(2)
            accuracy += 0.5 * (torch.log(2 * np.pi * spe_stds_b**2)  + ((next_spe  - spe_mus_b)  ** 2) / (spe_stds_b**2  + 1e-6)).sum(-1).unsqueeze(-1) 
        rgbd_complexity = self.args.beta_obs * dkl(rgbd_mus_b, rgbd_stds_b, torch.zeros(rgbd_mus_b.shape), self.args.sigma_obs * torch.ones(rgbd_stds_b.shape)).flatten(2)
        spe_complexity = self.args.beta_obs * dkl(spe_mus_b, spe_stds_b, torch.zeros(spe_mus_b.shape), self.args.sigma_obs * torch.ones(spe_stds_b.shape))
        zq_complexity  = self.args.beta_zq  * dkl(zq_mus,  zq_stds,  torch.zeros(zq_mus.shape),  self.args.sigma_zq  * torch.ones(zq_stds.shape))
                
        accuracy = accuracy * masks
        accuracy_loss = accuracy.mean()
        rgbd_complexity = rgbd_complexity * masks
        spe_complexity = spe_complexity * masks
        zq_complexity  = zq_complexity * masks
        obs_complexity_loss = (rgbd_complexity + spe_complexity).mean()
        zq_complexity_loss = zq_complexity.mean()
        forward_loss = accuracy_loss + obs_complexity_loss + zq_complexity_loss
        if(self.args.beta_obs == 0 and self.args.beta_zq == 0): obs_complexity_loss = None ; zq_complexity_loss = None
        
        zp_loss = dkl(zp_mus,  zp_stds, zq_mus,  zq_stds).mean()
        
        total_loss = forward_loss + zp_loss
        
        self.forward_opt.zero_grad()
        self.zp_opt.zero_grad()
        total_loss.backward()
        self.forward_opt.step()
        self.zp_opt.step()
        
                        
        
        # Get curiosity  
        naive_curiosity = self.args.naive_eta * accuracy.sum(-1).unsqueeze(-1)
        
        rgbd_mus_a = [] ; rgbd_stds_a = []
        spe_mus_a = [] ; spe_stds_a = [] ; h = None
        for step in range(steps):
            (_, rgbd_mu, rgbd_std), (_, spe_mu, spe_std), _, _, h = self.forward(rgbd[:, step], spe[:, step], prev_actions[:, step], actions[:, step], h)   
            rgbd_mus_a.append(rgbd_mu) ; rgbd_stds_a.append(rgbd_std)
            spe_mus_a.append(spe_mu) ; spe_stds_a.append(spe_std)
        rgbd_mus_a = torch.cat(rgbd_mus_a, dim = 1) ; rgbd_stds_a = torch.cat(rgbd_stds_a, dim = 1)
        spe_mus_a = torch.cat(spe_mus_a, dim = 1) ; spe_stds_a = torch.cat(spe_stds_a, dim = 1)
        
        state_dkl_changes = torch.clamp(dkl(zq_mus, zq_stds, zp_mus, zp_stds).sum(-1).unsqueeze(-1), min = 0, max = self.args.dkl_max)
        rgbd_dkl_changes = torch.clamp(dkl(rgbd_mus_a, rgbd_stds_a, rgbd_mus_b, rgbd_stds_b).flatten(2).sum(-1).unsqueeze(-1), min = 0, max = self.args.dkl_max)
        spe_dkl_changes  = torch.clamp(dkl(spe_mus_a, spe_stds_a, spe_mus_b, spe_stds_b).sum(-1).unsqueeze(-1), min = 0, max = self.args.dkl_max)
                    
        free_curiosity = self.args.free_eta_obs   * rgbd_dkl_changes  * masks + \
                         self.args.free_eta_obs   * spe_dkl_changes   * masks + \
                         self.args.free_eta_state * state_dkl_changes * masks
        
        if(self.args.curiosity == "naive"):  curiosity = naive_curiosity
        elif(self.args.curiosity == "free"): curiosity = free_curiosity
        else:                                curiosity = torch.zeros(rewards.shape)
        
        extrinsic = torch.mean(rewards).item()
        intrinsic_curiosity = curiosity.mean().item()
        rewards += curiosity
        
        
                
        # Train critics
        with torch.no_grad():
            next_action, log_pis_next, _ = self.actor(next_rgbd, next_spe, actions)
            Q_target1_next = self.critic1_target(next_rgbd, next_spe, actions, next_action)
            Q_target2_next = self.critic2_target(next_rgbd, next_spe, actions, next_action)
            Q_target_next = torch.min(Q_target1_next, Q_target2_next)
            if self.args.alpha == None: Q_targets = rewards + (self.args.GAMMA * (1 - dones) * (Q_target_next - self.alpha * log_pis_next))
            else:                       Q_targets = rewards + (self.args.GAMMA * (1 - dones) * (Q_target_next - self.args.alpha * log_pis_next))
        
        Q_1 = self.critic1(rgbd, spe, prev_actions, actions)
        critic1_loss = 0.5*F.mse_loss(Q_1*masks, Q_targets*masks)
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()
        
        Q_2 = self.critic2(rgbd, spe, prev_actions, actions)
        critic2_loss = 0.5*F.mse_loss(Q_2*masks, Q_targets*masks)
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()
        
        
        
        # Train alpha
        if self.args.alpha == None:
            _, log_pis, _ = self.actor(rgbd, spe, prev_actions)
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
            actions_pred, log_pis, _ = self.actor(rgbd, spe, prev_actions)

            if self.args.action_prior == "normal":
                loc = torch.zeros(action_size, dtype=torch.float64)
                scale_tril = torch.tensor([[1, 0], [1, 1]], dtype=torch.float64)
                policy_prior = MultivariateNormal(loc=loc, scale_tril=scale_tril)
                policy_prior_log_prrgbd = policy_prior.log_prob(actions_pred).unsqueeze(-1)
            elif self.args.action_prior == "uniform":
                policy_prior_log_prrgbd = 0.0
            Q = torch.min(
                self.critic1(rgbd, spe, prev_actions, actions_pred), 
                self.critic2(rgbd, spe, prev_actions, actions_pred)).mean(-1).unsqueeze(-1)
            intrinsic_entropy = torch.mean((alpha * log_pis)*masks).item()
            actor_loss = (alpha * log_pis - policy_prior_log_prrgbd - Q)*masks
            actor_loss = actor_loss.mean() / masks.mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.soft_update(self.critic1, self.critic1_target, self.args.tau)
            self.soft_update(self.critic2, self.critic2_target, self.args.tau)
            
        else:
            intrinsic_entropy = None
            actor_loss = None
        
        if(accuracy_loss != None):   accuracy_loss = accuracy_loss.item()
        if(obs_complexity_loss != None): obs_complexity_loss = obs_complexity_loss.item()
        if(zq_complexity_loss != None):  zq_complexity_loss = zq_complexity_loss.item()
        if(zp_loss != None): 
            zp_loss = zp_loss.item()
        if(alpha_loss != None): alpha_loss = alpha_loss.item()
        if(actor_loss != None): actor_loss = actor_loss.item()
        if(critic1_loss != None): 
            critic1_loss = critic1_loss.item()
            critic1_loss = log(critic1_loss) if critic1_loss > 0 else critic1_loss
        if(critic2_loss != None): 
            critic2_loss = critic2_loss.item()
            critic2_loss = log(critic2_loss) if critic2_loss > 0 else critic2_loss
        losses = np.array([[accuracy_loss, obs_complexity_loss, zq_complexity_loss, zp_loss, alpha_loss, actor_loss, critic1_loss, critic2_loss]])
        
        naive_curiosity = naive_curiosity.mean().item()
        free_curiosity = free_curiosity.mean().item()
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
