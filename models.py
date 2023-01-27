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
        
        self.weights = nn.Linear(2, args.hidden)
        self.bias = nn.Linear(2, args.hidden)
        self.dkl_out = nn.Linear(1 + 2*args.hidden, 1)
        
        self.weights.apply(init_weights)
        self.bias.apply(init_weights)
        self.dkl_out.apply(init_weights)
        self.to(args.device)
        
    def forward(self, errors, 
                before_w_mu, before_w_sigma, before_b_mu, before_b_sigma,
                after_w_mu, after_w_sigma, after_b_mu, after_b_sigma):
        
        change_w_mu  = after_w_mu - before_w_mu
        change_w_sigma = after_w_sigma - before_w_sigma
        weights = torch.cat([
            torch.cat([before_w_mu.unsqueeze(-1), change_w_mu.unsqueeze(-1)], dim = -2), 
            torch.cat([before_w_sigma.unsqueeze(-1), change_w_sigma.unsqueeze(-1)], dim = -2),], dim = -1)
        weights = self.weights(weights)
        weights = torch.mean(weights, 1, False)
        weights = weights.tile((1, errors.shape[1], errors.shape[2], 1))
        
        change_b_mu  = after_b_mu - before_b_mu
        change_b_sigma = after_b_sigma - before_b_sigma
        bias = torch.cat([
            torch.cat([before_b_mu.unsqueeze(-1), change_b_mu.unsqueeze(-1)], dim = -2), 
            torch.cat([before_b_sigma.unsqueeze(-1), change_b_sigma.unsqueeze(-1)], dim = -2),], dim = -1)
        bias = self.bias(bias)
        bias = torch.mean(bias, 1, False)
        bias = bias.tile((1, errors.shape[1], errors.shape[2], 1))
                
        dkl_out = self.dkl_out(torch.cat([errors, weights, bias], dim = -1))
        return(dkl_out)
    
        
        
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
    w_mu_shape    = (1, 8 * args.hidden + 6 * args.hidden)
    w_sigma_shape = (1, 8 * args.hidden + 6 * args.hidden)
    b_mu_shape    = (1, args.hidden + 6)
    b_sigma_shape = (1, args.hidden + 6)

    dkl_guesser = DKL_Guesser(args)

    print("\n\n")
    print(dkl_guesser)
    print()
    print(torch_summary(dkl_guesser, (
        errors_shape, 
        w_mu_shape, w_sigma_shape, b_mu_shape, b_sigma_shape,
        w_mu_shape, w_sigma_shape, b_mu_shape, b_sigma_shape)))
    


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
