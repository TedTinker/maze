#%% 

import torch
from torch import nn 
from torch.distributions import Normal
import torch.nn.functional as F
from torchinfo import summary as torch_summary

from math import exp

from utils import default_args, init_weights, invert_linear_layer
from maze import obs_size, action_size



class Summarizer(nn.Module): 
    
    def __init__(self, args = default_args):
        super(Summarizer, self).__init__()
        
        self.args = args
        self.gru = nn.GRU(
            input_size =  obs_size + action_size,
            hidden_size = args.hidden,
            batch_first = True)
        
        self.gru.apply(init_weights)
        self.to(args.device)
        
    def forward(self, obs, prev_a, h = None):
        x = torch.cat([obs, prev_a], -1)
        h = h if h == None else h.permute(1, 0, 2)
        h, _ = self.gru(x, h)
        return(h)
    
    

class Variational(nn.Module):
    
    def __init__(self, input_size, output_size, layers, std_min = exp(-20), std_max = exp(2), args = default_args):
        super(Variational, self).__init__()
        
        self.args = args
        self.std_min = std_min ; self.std_max = std_max
        
        self.mu = torch.nn.ModuleList()
        self.rho = torch.nn.ModuleList()
        for i in range(layers):
            in_size = args.hidden ; out_size = args.hidden
            if(i == 0): in_size = input_size
            if(i == layers - 1): out_size = output_size
            self.mu.append(nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.Tanh() if i != layers - 1 else nn.Identity()))
            self.rho.append(nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.LeakyReLU() if i != layers - 1 else nn.Identity()))
        
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
        x = torch.tanh(mu + e * std)
        def log_prob_func(x_, epsilon = 1e-6):
            log_prob = Normal(mu, std).log_prob(mu + e * std) - torch.log(1 - x_.pow(2) + epsilon)
            log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
            return(log_prob)
        return(x, mu, std, log_prob_func(x), log_prob_func)
        
        

class Forward(nn.Module):
    
    def __init__(self, args = default_args):
        super(Forward, self).__init__()
        
        self.args = args
        
        self.sum = Summarizer(args) 
        self.lin = nn.Linear(args.hidden + action_size, args.hidden)
        self.var = Variational(args.hidden, obs_size, args.forward_var_layers, args = args)
        
        self.lin.apply(init_weights)
        self.to(args.device)
        
    def forward(self, obs, prev_action, action):
        h = self.sum(obs, prev_action)
        x = torch.cat((h, action), dim=-1)
        x = self.lin(x)
        pred_obs, mu, std, _, log_prob_func = self.var(x)
        return(pred_obs, mu, std, log_prob_func)
        


class Actor(nn.Module):

    def __init__(self, args = default_args):
        super(Actor, self).__init__()
        
        self.args = args
        
        self.sum = Summarizer(args) 
        self.lin = nn.Sequential(
            nn.Linear(args.hidden, args.hidden),
            nn.LeakyReLU())
        self.var = Variational(args.hidden, action_size, args.actor_var_layers, args = args)

        self.lin.apply(init_weights)
        self.to(args.device)

    def forward(self, obs, prev_action, h = None):
        h = self.sum(obs, prev_action, h)
        x = self.lin(h)
        action, _, _, log_prob, _ = self.var(x)
        return(action, log_prob, h)
    
    
    
class Critic(nn.Module):

    def __init__(self, args = default_args):
        super(Critic, self).__init__()
        
        self.args = args
        
        self.sum = Summarizer(args) 
        self.lin = nn.Sequential(
            nn.Linear(args.hidden + action_size, args.hidden),
            nn.LeakyReLU(),
            nn.Linear(args.hidden, 1))

        self.lin.apply(init_weights)
        self.to(args.device)

    def forward(self, obs, prev_action, action):
        h = self.sum(obs, prev_action)
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
    print(torch_summary(forward, ((3, obs_size), (3, action_size))))
    


    actor = Actor(args)
    
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor, ((3,obs_size),)))
    
    
    
    critic = Critic(args)
    
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, ((3,obs_size),(3,action_size),(3,action_size))))

# %%
