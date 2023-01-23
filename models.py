#%% 

import torch
from torch import nn 
from torch.distributions import Normal
from torchinfo import summary as torch_summary
from blitz.modules import BayesianLinear

from utils import device, default_args, init_weights



class Forward(nn.Module):
    
    def __init__(self, args):
        super(Forward, self).__init__()
        
        self.pos_out = nn.Sequential(
            nn.Linear(8, args.hidden),
            #nn.Linear(args.hidden, args.hidden),
            nn.Linear(args.hidden, 6))
        
        self.pos_out.apply(init_weights)
        self.to(device)
        
    def forward(self, pos, action):
        pos = pos.to(device) ; action = action.to(device)
        x = torch.cat([pos, action], -1)
        x = self.pos_out(x).to("cpu")
        return(x) 
    
    
    
class Bayes_Forward(nn.Module):
    
    def __init__(self, args):
        super(Bayes_Forward, self).__init__()
        
        self.pos_out = nn.Sequential(
            BayesianLinear(8, args.hidden),
            #BayesianLinear(args.hidden, args.hidden),
            BayesianLinear(args.hidden, 6))
        
        self.pos_out.apply(init_weights)
        self.to(device)
        
    def forward(self, pos, action):
        pos = pos.to(device) ; action = action.to(device)
        x = torch.cat([pos, action], -1)
        x = self.pos_out(x).to("cpu")
        return(x) 
        
        
        
class Actor(nn.Module):

    def __init__(self, args, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()

        self.log_std_min = log_std_min ; self.log_std_max = log_std_max
        self.lin = nn.Sequential(
            nn.Linear(6, args.hidden),
            nn.LeakyReLU(),)
            #nn.Linear(args.hidden, args.hidden),
            #nn.LeakyReLU())
        self.mu = nn.Linear(args.hidden, 2)
        self.log_std_linear = nn.Linear(args.hidden, 2)

        self.lin.apply(init_weights)
        self.mu.apply(init_weights)
        self.log_std_linear.apply(init_weights)
        self.to(device)

    def forward(self, pos):
        pos = pos.to(device)
        x = self.lin(pos)
        mu = self.mu(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def evaluate(self, pos, epsilon=1e-6):
        mu, log_std = self.forward(pos)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample(std.shape).to(device)
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - \
            torch.log(1 - action.pow(2) + epsilon)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        return action, log_prob

    def get_action(self, pos):
        mu, log_std = self.forward(pos)
        std = log_std.exp()
        dist = Normal(0, 1)
        e      = dist.sample(std.shape).to(device)
        action = torch.tanh(mu + e * std).cpu()
        return action[0]
    
    
    
class Critic(nn.Module):

    def __init__(self, args):
        super(Critic, self).__init__()
                
        self.lin = nn.Sequential(
            nn.Linear(8, args.hidden),
            nn.LeakyReLU(),
            #nn.Linear(args.hidden, args.hidden),
            #nn.LeakyReLU(),
            nn.Linear(args.hidden, 1))

        self.lin.apply(init_weights)
        self.to(device)

    def forward(self, pos, action):
        pos = pos.to(device) ; action = action.to(device)
        x = torch.cat((pos, action), dim=-1)
        x = self.lin(x).to("cpu")
        return x
    


if __name__ == "__main__":

    forward = Forward(default_args)
    
    print("\n\n")
    print(forward)
    print()
    print(torch_summary(forward, ((1,6), (1,2))))
    
    
    
    bayes_forward = Bayes_Forward(default_args)
    
    print("\n\n")
    print(bayes_forward)
    print()
    print(torch_summary(bayes_forward, ((1,6), (1,2))))
    


    actor = Actor(default_args)
    
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor, ((1,6),)))
    
    
    
    critic = Critic(default_args)
    
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, ((1,6),(1,2))))
# %%
