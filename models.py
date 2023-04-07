#%% 

import torch
from torch import nn 
from torch.distributions import Normal
import torch.nn.functional as F
from torchinfo import summary as torch_summary

from math import exp

from utils import default_args, init_weights
from maze import obs_size, action_size



class Variational(nn.Module):
    
    def __init__(self, input_size, output_size, layers, std_min = exp(-20), std_max = exp(2), args = default_args):
        super(Variational, self).__init__()
        
        self.args = args
        self.std_min = std_min ; self.std_max = std_max
        
        self.mu = torch.nn.ModuleList()
        self.rho = torch.nn.ModuleList()
        for i in range(layers):
            in_size = args.hidden_size ; out_size = args.hidden_size
            if(i == 0): in_size = input_size
            if(i == layers - 1): out_size = output_size
            self.mu.append(nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.Tanh() if i != layers - 1 else nn.Identity()))
            self.rho.append(nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.Tanh() if i != layers - 1 else nn.Identity()))
        
        self.mu.apply(init_weights)
        self.rho.apply(init_weights)
        self.to(args.device)
        
    def forward(self, x):
        for i, (mu_layer, rho_layer) in enumerate(zip(self.mu, self.rho)):
            if(i == 0): mu = mu_layer(x)  ; rho = rho_layer(x)
            else:       mu = mu_layer(mu) ; rho = rho_layer(rho)
        std = torch.log1p(torch.exp(rho))
        std = torch.clamp(std, min = self.std_min, max = self.std_max)
        e = Normal(0, 1).sample(std.shape).to("cuda" if next(self.parameters()).is_cuda else "cpu")
        x = mu + e * std
        return(x, mu, std)
    
        
        
class Forward(nn.Module):
    
    def __init__(self, args = default_args):
        super(Forward, self).__init__()
        
        self.args = args
        
        self.gru = nn.GRU(
            input_size =  args.hidden_size,
            hidden_size = args.hidden_size,
            batch_first = True)
        self.zq_var  = Variational(args.hidden_size + obs_size + action_size, args.state_size, args.state_var_layers, args = args)
        self.obs_var = Variational(args.hidden_size + action_size,            obs_size,        args.obs_var_layers, args = args)
        
        self.gru.apply(init_weights)
        self.to(args.device)
        
    def zq(self, obs, prev_action, h = None):
        if(len(obs.shape) == 2): obs = obs.unsqueeze(1)
        if(len(prev_action.shape) == 2): prev_action = prev_action.unsqueeze(1)
        if(h == None): h = torch.zeros((obs.shape[0], 1, self.args.hidden_size)).to(obs.device)
        x = torch.cat((h, obs, prev_action), dim=-1)
        zq, zq_mu, zq_std = self.zq_var(x)
        zq = torch.tanh(zq)
        h = h if h == None else h.permute(1, 0, 2)
        h, _ = self.gru(zq, h)
        return(zq, zq_mu, zq_std, h)
        
    def forward(self, obs, prev_action, action, h = None):
        zq, zq_mu, zq_std, h = self.zq(obs, prev_action, h)
        x = torch.cat((h, action.unsqueeze(1)), dim=-1)
        pred_obs, obs_mu, obs_std = self.obs_var(x)
        #pred_obs = torch.clamp(pred_obs, min = -1, max = 1)
        pred_obs = torch.tanh(pred_obs)
        return(pred_obs, obs_mu, obs_std, zq, zq_mu, zq_std, h)
        


class Actor(nn.Module):

    def __init__(self, args = default_args):
        super(Actor, self).__init__()
        
        self.args = args
        
        self.gru = nn.GRU(
            input_size =  obs_size + action_size,
            hidden_size = args.hidden_size,
            batch_first = True)
        self.var = Variational(args.hidden_size, action_size, args.actor_var_layers, args = args)

        self.gru.apply(init_weights)
        self.to(args.device)

    def forward(self, obs, prev_action, h = None):
        x = torch.cat((obs, prev_action), dim=-1)
        h, _ = self.gru(x, h)
        x, mu, std = self.var(h)
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
