#%% 

import torch
from torch import nn 
from torch.distributions import Normal
from torchinfo import summary as torch_summary

from utils import default_args, init_weights
from easy_maze import obs_size, action_size
    
        

def var(x, mu_func, rho_func, args):
    mu = mu_func(x)
    std = torch.log1p(torch.exp(rho_func(x)))
    std = torch.clamp(std, min = args.std_min, max = args.std_max)
    return(mu, std)

def sample(mu, std):
    e = Normal(0, 1).sample(std.shape).to("cuda" if std.is_cuda else "cpu")
    return(mu + e * std)
    
    
    
class Forward(nn.Module):
    
    def __init__(self, args = default_args):
        super(Forward, self).__init__()
        
        self.args = args
        
        self.gru = nn.GRU(
            input_size =  args.hidden_size,
            hidden_size = args.hidden_size,
            batch_first = True)
        
        self.zp_mu = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.state_size), 
            nn.Tanh())
        self.zp_rho = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.state_size))
        
        self.zq_mu = nn.Sequential(
            nn.Linear(args.hidden_size + obs_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.state_size), 
            nn.Tanh())
        self.zq_rho = nn.Sequential(
            nn.Linear(args.hidden_size + obs_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.state_size))
        
        self.obs = nn.Sequential(
            nn.Linear(args.hidden_size + action_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, obs_size), 
            nn.Tanh())
        
        self.gru.apply(init_weights)
        self.zp_mu.apply(init_weights)
        self.zp_rho.apply(init_weights)
        self.zq_mu.apply(init_weights)
        self.zq_rho.apply(init_weights)
        self.obs.apply(init_weights)
        self.to(args.device)
        
    def forward(self, obs, action, h = None, quantity = 1):
        if(len(obs.shape) == 2): obs = obs.unsqueeze(1)
        if(len(action.shape) == 2): action = action.unsqueeze(1)
        if(h == None): h = torch.zeros((obs.shape[0], 1, self.args.hidden_size)).to(obs.device)
        zp_mu, zp_std = var(h,                           self.zp_mu, self.zp_rho, self.args)
        zq_mu, zq_std = var(torch.cat((h, obs), dim=-1), self.zq_mu, self.zq_rho, self.args)
        
        h = h.permute(1, 0, 2)
        
        zp_h, _ = self.gru(zp_mu, h)
        zp_x = torch.cat((zp_h, action), dim=-1)
        zp_mu_pred = self.obs(zp_x)
        
        zq_h, _ = self.gru(zq_mu, h)
        zq_x = torch.cat((zq_h, action), dim=-1)
        zq_mu_pred = self.obs(zq_x)
        
        zp_pred_obs = [] ; zq_pred_obs = []
        for _ in range(quantity):
            zp = sample(zp_mu, zp_std)
            zp_h, _ = self.gru(zp, h)
            zp_pred_obs.append(self.obs(torch.cat((zp_h, action), dim=-1)))
            
            zq = sample(zq_mu, zq_std)
            zq_h, _ = self.gru(zq, h)
            zq_pred_obs.append(self.obs(torch.cat((zq_h, action), dim=-1)))
        if(quantity == 0): 
            zq = sample(zq_mu, zq_std)
            zq_h, _ = self.gru(zq, h)
        
        return((zp_mu_pred, zp_pred_obs), (zq_mu_pred, zq_pred_obs), (zp, zp_mu, zp_std), (zq, zq_mu, zq_std), zq_h)
        


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
        mu, std = var(h, self.mu, self.rho, self.args)
        x = sample(mu, std)
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
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, 1))

        self.gru.apply(init_weights)
        self.lin.apply(init_weights)
        self.to(args.device)

    def forward(self, obs, action, h = None):
        x = torch.cat((obs, action), dim=-1)
        h, _ = self.gru(x, h)
        Q = self.lin(h)
        return(Q)
    


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
