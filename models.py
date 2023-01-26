#%% 

import torch
from torch import nn 
from torch.distributions import Normal
from torchinfo import summary as torch_summary
from blitz.modules import BayesianLinear

from utils import init_weights, get_title

    
    
class Forward(nn.Module):
    
    def __init__(self, args):
        super(Forward, self).__init__()
        
        self.args = args
        
        self.pos_out = nn.Sequential(
            BayesianLinear(8, args.hidden, bias = args.bias),
            BayesianLinear(args.hidden, 6, bias = args.bias))
        
        self.pos_out.apply(init_weights)
        self.to(args.device)
        
    def forward(self, pos, action):
        pos = pos.to(self.args.device) ; action = action.to(self.args.device)
        x = torch.cat([pos, action], -1)
        x = self.pos_out(x).to("cpu")
        return(x) 
    
    
    
class DKL_Guesser(nn.Module):
    
    def __init__(self, args):
        super(DKL_Guesser, self).__init__()
        
        self.error_in = nn.Linear(1, args.dkl_hidden)
        self.w_mu     = nn.Linear(args.hidden * 8 + args.hidden * 6, args.dkl_hidden)
        self.w_sigma  = nn.Linear(args.hidden * 8 + args.hidden * 6, args.dkl_hidden)
        self.b_mu     = nn.Linear(args.hidden + 6, args.dkl_hidden)
        self.b_sigma  = nn.Linear(args.hidden + 6, args.dkl_hidden)
        self.DKL_out  = nn.Linear(args.dkl_hidden * 5, 1)
        
        self.error_in.apply(init_weights)
        self.w_mu.apply(init_weights)
        self.w_sigma.apply(init_weights)
        self.b_mu.apply(init_weights)
        self.b_sigma.apply(init_weights)
        self.to(args.device)
        
    def forward(self, errors, weights_mu, weights_sigma, bias_mu, bias_sigma):
        errors  = self.error_in(errors) 
        w_mu    = self.w_mu(weights_mu)
        w_sigma = self.w_sigma(weights_sigma)
        b_mu    = self.b_mu(bias_mu)
        b_sigma = self.b_sigma(bias_sigma) 
        w_mu    = torch.tile(w_mu, (1, errors.shape[1], errors.shape[2], 1))
        w_sigma = torch.tile(w_sigma, (1, errors.shape[1], errors.shape[2], 1))
        b_mu    = torch.tile(b_mu, (1, errors.shape[1], errors.shape[2], 1))
        b_sigma = torch.tile(b_sigma, (1, errors.shape[1], errors.shape[2], 1))
        x = torch.cat([errors, w_mu, w_sigma, b_mu, b_sigma], -1)
        x = self.DKL_out(x).to("cpu")
        return(x)
    
        
        
class Actor(nn.Module):

    def __init__(self, args, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        
        self.args = args

        self.log_std_min = log_std_min ; self.log_std_max = log_std_max
        self.lin = nn.Sequential(
            nn.Linear(6, args.hidden),
            nn.LeakyReLU())
        self.mu = nn.Linear(args.hidden, 2)
        self.log_std_linear = nn.Linear(args.hidden, 2)

        self.lin.apply(init_weights)
        self.mu.apply(init_weights)
        self.log_std_linear.apply(init_weights)
        self.to(self.args.device)

    def forward(self, pos):
        pos = pos.to(self.args.device)
        x = self.lin(pos)
        mu = self.mu(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def evaluate(self, pos, epsilon=1e-6):
        mu, log_std = self.forward(pos)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample(std.shape).to(self.args.device)
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - \
            torch.log(1 - action.pow(2) + epsilon)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        return action, log_prob

    def get_action(self, pos):
        mu, log_std = self.forward(pos)
        std = log_std.exp()
        dist = Normal(0, 1)
        e      = dist.sample(std.shape).to(self.args.device)
        action = torch.tanh(mu + e * std).cpu()
        return action[0]
    
    
    
class Critic(nn.Module):

    def __init__(self, args):
        super(Critic, self).__init__()
        
        self.args = args
                
        self.lin = nn.Sequential(
            nn.Linear(8, args.hidden),
            nn.LeakyReLU(),
            nn.Linear(args.hidden, 1))

        self.lin.apply(init_weights)
        self.to(args.device)

    def forward(self, pos, action):
        pos = pos.to(self.args.device) ; action = action.to(self.args.device)
        x = torch.cat((pos, action), dim=-1)
        x = self.lin(x).to("cpu")
        return x
    


if __name__ == "__main__":
    
    args, _ = get_title({"device" : "cuda"})

    forward = Forward(args)
    
    print("\n\n")
    print(forward)
    print()
    print(torch_summary(forward, ((1,6), (1,2))))
    
    
    
    errors_shape  = (1, 8, 10, 1)
    w_mu_shape    = (1, 1, 1, 448)
    w_sigma_shape = (1, 1, 1, 448)
    b_mu_shape    = (1, 1, 1, 38)
    b_sigma_shape = (1, 1, 1, 38)

    dkl_guesser = DKL_Guesser(args)

    print("\n\n")
    print(dkl_guesser)
    print()
    print(torch_summary(dkl_guesser, (errors_shape, w_mu_shape, w_sigma_shape, b_mu_shape, b_sigma_shape)))
    


    actor = Actor(args)
    
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor, ((1,6),)))
    
    
    
    critic = Critic(args)
    
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, ((1,6),(1,2))))
# %%
