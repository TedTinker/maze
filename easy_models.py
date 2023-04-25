#%% 

import torch
from torch import nn 
from torch.distributions import Normal
from torchinfo import summary as torch_summary

from utils import default_args, init_weights, print
from easy_maze import obs_size, action_size
    
        

def var(x, mu_func, std_func, args):
    mu = mu_func(x)
    std = torch.clamp(std_func(x), min = args.std_min, max = args.std_max)
    return(mu, std)

def sample(mu, std):
    e = Normal(0, 1).sample(std.shape).to("cuda" if std.is_cuda else "cpu")
    return(mu + e * std)
    
    
    
class Forward(nn.Module):
    
    def __init__(self, args = default_args):
        super(Forward, self).__init__()
        
        self.args = args
        
        self.zp_mu = nn.Sequential(
            nn.Linear(args.hidden_size + action_size, args.hidden_size), 
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, args.state_size), 
            nn.Tanh())
        self.zp_std = nn.Sequential(
            nn.Linear(args.hidden_size + action_size, args.hidden_size), 
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, args.state_size),
            nn.Softplus())
        
        self.zq_mu = nn.Sequential(
            nn.Linear(args.hidden_size + action_size + obs_size, args.hidden_size), 
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, args.state_size), 
            nn.Tanh())
        self.zq_std = nn.Sequential(
            nn.Linear(args.hidden_size + action_size + obs_size, args.hidden_size), 
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, args.state_size),
            nn.Softplus())
        
        self.gru = nn.GRU(
            input_size =  args.state_size,
            hidden_size = args.hidden_size,
            batch_first = True)
        
        self.obs = nn.Sequential(
            nn.Linear(args.hidden_size + action_size, args.hidden_size), 
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, obs_size), 
            nn.Tanh())
        
        self.zp_mu.apply(init_weights)
        self.zp_std.apply(init_weights)
        self.zq_mu.apply(init_weights)
        self.zq_std.apply(init_weights)
        self.gru.apply(init_weights)
        self.obs.apply(init_weights)
        self.to(args.device)
        
    def forward(self, obs, prev_a, h_q_m1):
        if(len(obs.shape) == 2):    obs    = obs.unsqueeze(1)
        if(len(prev_a.shape) == 2): prev_a = prev_a.unsqueeze(1)
        zp_mu, zp_std = var(torch.cat((h_q_m1, prev_a), dim=-1),      self.zp_mu, self.zp_std, self.args)
        zq_mu, zq_std = var(torch.cat((h_q_m1, prev_a, obs), dim=-1), self.zq_mu, self.zq_std, self.args)        
        zq = sample(zq_mu, zq_std)
        h_q, _ = self.gru(zq, h_q_m1.permute(1, 0, 2))
        return((zp_mu, zp_std), (zq_mu, zq_std), h_q)
    
    def get_preds(self, action, z_mu, z_std, h_q_m1, quantity = 1):
        if(len(action.shape) == 2): action = action.unsqueeze(1)
        h_q_m1 = h_q_m1.permute(1, 0, 2)
        
        h, _ = self.gru(z_mu, h_q_m1)
        mu_pred = self.obs(torch.cat((h, action), dim=-1))
            
        pred_obs = []
        for _ in range(quantity):
            z = sample(z_mu, z_std)
            h, _ = self.gru(z, h_q_m1)
            pred_obs.append(self.obs(torch.cat((h, action), dim=-1)))
        return(mu_pred, pred_obs)
        


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
        self.std = nn.Sequential(
            nn.Linear(args.hidden_size, action_size),
            nn.Softplus())

        self.gru.apply(init_weights)
        self.mu.apply(init_weights)
        self.std.apply(init_weights)
        self.to(args.device)

    def forward(self, obs, prev_action, h = None):
        h, _ = self.gru(torch.cat((obs, prev_action), dim=-1), h)
        mu, std = var(h, self.mu, self.std, self.args)
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
        h, _ = self.gru(torch.cat((obs, action), dim=-1), h)
        Q = self.lin(h)
        return(Q)
    
    
    
class Actor_HQ(nn.Module):

    def __init__(self, args = default_args):
        super(Actor, self).__init__()
        
        self.args = args
        
        self.lin = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.LeakyReLU())
        self.mu = nn.Sequential(
            nn.Linear(args.hidden_size, action_size))
        self.std = nn.Sequential(
            nn.Linear(args.hidden_size, action_size),
            nn.Softplus())

        self.lin.apply(init_weights)
        self.mu.apply(init_weights)
        self.std.apply(init_weights)
        self.to(args.device)

    def forward(self, h):
        x = self.lin(h)
        mu, std = var(x, self.mu, self.std, self.args)
        x = sample(mu, std)
        #action = torch.clamp(x, min = -1, max = 1)
        action = torch.tanh(x)
        log_prob = Normal(mu, std).log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        return(action, log_prob)
    
    
    
class Critic_HQ(nn.Module):

    def __init__(self, args = default_args):
        super(Critic, self).__init__()
        
        self.args = args
        
        self.lin = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, 1))

        self.lin.apply(init_weights)
        self.to(args.device)

    def forward(self, h, action):
        Q = self.lin(torch.cat((h, action), dim=-1))
        return(Q)
    


if __name__ == "__main__":
    
    args = default_args
    args.device = "cpu"
    args.dkl_rate = 1
    
    
    
    forward = Forward(args)
    
    print("\n\n")
    print(forward)
    print()
    print(torch_summary(forward, ((3, 1, obs_size), (3, 1, args.hidden_size))))
    


    actor = Actor(args)
    
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor, ((3, 1, obs_size), (3, 1, action_size))))
    
    
    
    critic = Critic(args)
    
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, ((3, 1, obs_size), (3, 1, action_size))))

# %%
