#%% 

import torch
from torch import nn 
from torch.distributions import Normal
import torch.nn.functional as F
from torchinfo import summary as torch_summary

from math import exp

from utils import default_args, init_weights
from maze import obs_size, action_size
    
        

class State_Forward(nn.Module):
    
    def __init__(self, args = default_args):
        super(State_Forward, self).__init__()
        
        self.args = args
        
        self.gru = nn.GRU(
            input_size =  args.hidden_size,
            hidden_size = args.hidden_size,
            batch_first = True)
        self.zq_mu = nn.Sequential(
            nn.Linear(args.hidden_size + obs_size + action_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.state_size))
        self.zq_rho = nn.Sequential(
            nn.Linear(args.hidden_size + obs_size + action_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.state_size))
        self.obs_mu = nn.Sequential(
            nn.Linear(args.hidden_size + action_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, obs_size), 
            nn.Tanh())
        self.obs_rho = nn.Sequential(
            nn.Linear(args.hidden_size + action_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, obs_size))
        
        self.gru.apply(init_weights)
        self.zq_mu.apply(init_weights)
        self.zq_rho.apply(init_weights)
        self.obs_mu.apply(init_weights)
        self.obs_rho.apply(init_weights)
        self.to(args.device)
        
    def zq(self, obs, prev_action, h = None):
        if(len(obs.shape) == 2): obs = obs.unsqueeze(1)
        if(len(prev_action.shape) == 2): prev_action = prev_action.unsqueeze(1)
        if(h == None): h = torch.zeros((obs.shape[0], 1, self.args.hidden_size)).to(obs.device)
        x = torch.cat((h, obs, prev_action), dim=-1)
        zq_mu = self.zq_mu(x)
        zq_std = torch.log1p(torch.exp(self.zq_rho(x)))
        zq_std = torch.clamp(zq_std, min = self.args.std_min, max = self.args.std_max)
        e = Normal(0, 1).sample(zq_std.shape).to("cuda" if next(self.parameters()).is_cuda else "cpu")
        zq = zq_mu + e * zq_std
        h = h if h == None else h.permute(1, 0, 2)
        h, _ = self.gru(zq, h)
        return(zq, zq_mu, zq_std, h)
        
    def forward(self, obs, prev_action, action, h = None):
        zq, zq_mu, zq_std, h = self.zq(obs, prev_action, h)
        x = torch.cat((h, action.unsqueeze(1)), dim=-1)
        obs_mu = self.obs_mu(x)
        obs_std = torch.log1p(torch.exp(self.obs_rho(x)))
        obs_std = torch.clamp(obs_std, min = self.args.std_min, max = self.args.std_max)
        e = Normal(0, 1).sample(obs_std.shape).to("cuda" if next(self.parameters()).is_cuda else "cpu")
        pred_obs = obs_mu + e * obs_std
        #pred_obs = torch.clamp(pred_obs, min = -1, max = 1)
        return(pred_obs, obs_mu, obs_std, zq, zq_mu, zq_std, h)



class Forward(nn.Module):
    
    def __init__(self, args = default_args):
        super(Forward, self).__init__()
        
        self.args = args
        
        self.gru = nn.GRU(
            input_size =  obs_size + action_size,
            hidden_size = args.hidden_size,
            batch_first = True)
        self.obs_mu = nn.Sequential(
            nn.Linear(args.hidden_size + action_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, obs_size), 
            nn.Tanh())
        self.obs_rho = nn.Sequential(
            nn.Linear(args.hidden_size + action_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, obs_size))
        
        self.gru.apply(init_weights)
        self.obs_mu.apply(init_weights)
        self.obs_rho.apply(init_weights)
        self.to(args.device)
        
    def forward(self, obs, prev_action, action, h = None):
        x = torch.cat((obs, prev_action), dim=-1)
        h, _ = self.gru(x, h)
        x = torch.cat((h, action), dim=-1)
        obs_mu = self.obs_mu(x)
        obs_std = torch.log1p(torch.exp(self.obs_rho(x)))
        obs_std = torch.clamp(obs_std, min = self.args.std_min, max = self.args.std_max)
        e = Normal(0, 1).sample(obs_std.shape).to("cuda" if next(self.parameters()).is_cuda else "cpu")
        pred_obs = obs_mu + e * obs_std
        #pred_obs = torch.clamp(pred_obs, min = -1, max = 1)
        return(pred_obs, obs_mu, obs_std)
        


class Actor(nn.Module):

    def __init__(self, args = default_args):
        super(Actor, self).__init__()
        
        self.args = args
        
        self.gru = nn.GRU(
            input_size =  obs_size + action_size,
            hidden_size = args.hidden_size,
            batch_first = True)
        self.mu = nn.Sequential(
            nn.Linear(args.hidden_size, action_size))
        self.rho = nn.Sequential(
            nn.Linear(args.hidden_size, action_size))

        self.gru.apply(init_weights)
        self.mu.apply(init_weights)
        self.rho.apply(init_weights)
        self.to(args.device)

    def forward(self, obs, prev_action, h = None):
        x = torch.cat((obs, prev_action), dim=-1)
        h, _ = self.gru(x, h)
        mu = self.mu(h)
        std = torch.log1p(torch.exp(self.rho(h)))
        std = torch.clamp(std, min = self.args.std_min, max = self.args.std_max)
        e = Normal(0, 1).sample(std.shape).to("cuda" if next(self.parameters()).is_cuda else "cpu")
        x = mu + e * std
        #action = torch.clamp(x, min = -1, max = 1)
        action = torch.tanh(x)
        log_prob = Normal(mu, std).log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        return(action, log_prob, h)
    
    
    
class Critic(nn.Module):

    def __init__(self, args = default_args):
        super(Critic, self).__init__()
        
        self.args = args
        
        self.gru = nn.GRU(
            input_size =  obs_size + action_size,
            hidden_size = args.hidden_size,
            batch_first = True)
        self.lin = nn.Sequential(
            nn.Linear(args.hidden_size + action_size, args.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, 1))

        self.gru.apply(init_weights)
        self.lin.apply(init_weights)
        self.to(args.device)

    def forward(self, obs, prev_action, action, h = None):
        x = torch.cat((obs, prev_action), dim=-1)
        h, _ = self.gru(x, h)
        x = torch.cat((h, action), dim=-1)
        x = self.lin(x)
        return(x)
    


if __name__ == "__main__":
    
    args = default_args
    args.device = "cpu"
    args.dkl_rate = 1
    
    forward = State_Forward(args)
    
    print("\n\n")
    print(forward)
    print()
    print(torch_summary(forward, ((3, obs_size), (3, action_size), (3, action_size))))
    
    
    
    forward = Forward(args)
    
    print("\n\n")
    print(forward)
    print()
    print(torch_summary(forward, ((3, obs_size), (3, action_size), (3, action_size))))
    


    actor = Actor(args)
    
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor, ((3, 1, obs_size), (3, 1, action_size))))
    
    
    
    critic = Critic(args)
    
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, ((3, 1, obs_size), (3, 1, action_size), (3, 1, action_size))))

# %%
