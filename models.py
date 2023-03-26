#%% 

import torch
from torch import nn 
from torch.distributions import Normal
import torch.nn.functional as F
from torchinfo import summary as torch_summary

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
        self.to(self.args.device)
        
    def forward(self, obs, prev_a, h = None):
        x = torch.cat([obs, prev_a], -1)
        h = h if h == None else h.permute(1, 0, 2)
        h, _ = self.gru(x, h)
        return(h)
    
    

# I think this needs min-max!
class Variational(nn.Module):
    
    def __init__(self, input_size, output_size, args = default_args):
        super(Variational, self).__init__()
        
        self.args = args
        
        self.mu = nn.Linear(input_size, output_size)
        self.rho = nn.Linear(input_size, output_size)
        
        self.mu.apply(init_weights)
        self.rho.apply(init_weights)
        self.to(args.device)
        
    def forward(self, x):
        mu = self.mu(x)
        std = torch.log1p(torch.exp(self.rho(x))) 
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
        
        self.sum = Summarizer(self.args) # Not implemented
        self.lin = nn.Linear(obs_size + action_size, args.hidden)
        self.var = Variational(args.hidden, args.hidden, args = args)
        self.lin_2 = nn.Linear(args.hidden, obs_size)
        
        self.lin.apply(init_weights)
        self.to(args.device)
        
    def forward(self, obs, action):
        x = torch.cat([obs, action], -1)
        x = self.lin(x)
        x, mu, std, _, log_prob_func = self.var(x)
        pred_obs = self.lin_2(x)
        reverse = invert_linear_layer(self.lin_2)
        def new_log_prob_func(pred_obs, epsilon = 1e-6):
            x = reverse(pred_obs)
            return(log_prob_func(x, epsilon))
        return(pred_obs, mu, std, new_log_prob_func)
        


class Actor(nn.Module):

    def __init__(self, args = default_args):
        super(Actor, self).__init__()
        
        self.args = args
        
        self.sum = Summarizer(self.args) # Not implemented
        self.lin = nn.Sequential(
            nn.Linear(obs_size, args.hidden),
            nn.LeakyReLU())
        self.var = Variational(args.hidden, action_size, args = args)

        self.lin.apply(init_weights)
        self.to(self.args.device)

    def forward(self, obs):
        x = self.lin(obs)
        action, _, _, log_prob, _ = self.var(x)
        return(action, log_prob)
    
    
    
class Critic(nn.Module):

    def __init__(self, args = default_args):
        super(Critic, self).__init__()
        
        self.args = args
        
        self.sum = Summarizer(self.args) # Implemented
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
