#%% 

import torch
from torch import nn 
from torch.distributions import Normal
from torchinfo import summary as torch_summary

from utils import default_args, init_weights, ConstrainedConv2d, print
spe_size = 1 ; action_size = 2



def var(x, mu_func, std_func, args):
    mu = mu_func(x)
    std = torch.clamp(std_func(x), min = args.std_min, max = args.std_max)
    return(mu, std)

def sample(mu, std):
    e = Normal(0, 1).sample(std.shape).to("cuda" if std.is_cuda else "cpu")
    return(mu + e * std)

def rnn_cnn(do_this, to_this):
    episodes = to_this.shape[0] ; steps = to_this.shape[1]
    this = to_this.view((episodes * steps, to_this.shape[2], to_this.shape[3], to_this.shape[4]))
    this = do_this(this)
    this = this.view((episodes, steps, this.shape[1], this.shape[2], this.shape[3]))
    return(this)

        

class Forward(nn.Module):
    
    def __init__(self, args = default_args):
        super(Forward, self).__init__()
        
        self.args = args
        rgbd_size = (1, 4, args.image_size, args.image_size)
        example = torch.zeros(rgbd_size)
        
        self.rgbd_in = nn.Sequential(
            ConstrainedConv2d(
                in_channels = 4,
                out_channels = 4,
                kernel_size = (3,3),
                padding = (1,1),
                padding_mode = "reflect"),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size = (3,3), 
                stride = (2,2),
                padding = (1,1)))
        example = self.rgbd_in(example).flatten(1)
        rgbd_size = example.shape[1]
        
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
            nn.Linear(args.hidden_size + action_size + rgbd_size + spe_size, args.hidden_size), 
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, args.state_size),
            nn.Tanh())
        self.zq_std = nn.Sequential(
            nn.Linear(args.hidden_size + action_size + rgbd_size + spe_size, args.hidden_size), 
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, args.state_size),
            nn.Softplus())
        
        self.gru = nn.GRU(
            input_size =  args.state_size,
            hidden_size = args.hidden_size,
            batch_first = True)
        
        self.rgbd_up = nn.Sequential(
            nn.Linear(args.hidden_size + action_size, rgbd_size),
            nn.LeakyReLU())
        self.rgbd = nn.Sequential(
            ConstrainedConv2d(
                in_channels = 4,
                out_channels = 4,
                kernel_size = (3,3),
                padding = (1,1),
                padding_mode = "reflect"),
            nn.LeakyReLU(),
            nn.Upsample(
                scale_factor = 2, 
                mode = "bilinear",
                align_corners = True),
            ConstrainedConv2d(
                in_channels = 4,
                out_channels = 4,
                kernel_size = (1,1)))
        
        self.spe = nn.Sequential(
            nn.Linear(args.hidden_size + action_size, args.hidden_size), 
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, spe_size))
        
        self.rgbd_in.apply(init_weights)
        self.zp_mu.apply(init_weights)
        self.zp_std.apply(init_weights)
        self.zq_mu.apply(init_weights)
        self.zq_std.apply(init_weights)
        self.gru.apply(init_weights)
        self.rgbd_up.apply(init_weights)
        self.rgbd.apply(init_weights)
        self.spe.apply(init_weights)
        self.to(args.device)
        
    def forward(self, rgbd, spe, prev_a, h_q_m1):
        if(len(rgbd.shape) == 4):    rgbd   = rgbd.unsqueeze(1)
        if(len(spe.shape) == 2):     spe    = spe.unsqueeze(1)
        if(len(prev_a.shape) == 2):  prev_a = prev_a.unsqueeze(1)
        rgbd = (rgbd * 2) - 1
        spe = (spe - self.args.min_speed) / (self.args.max_speed - self.args.min_speed)
        rgbd = rnn_cnn(self.rgbd_in, rgbd.permute(0, 1, 4, 2, 3)).flatten(2)
        zp_mu, zp_std = var(torch.cat((h_q_m1, prev_a),            dim=-1), self.zp_mu, self.zp_std, self.args)
        zq_mu, zq_std = var(torch.cat((h_q_m1, prev_a, rgbd, spe), dim=-1), self.zq_mu, self.zq_std, self.args)        
        zq = sample(zq_mu, zq_std)
        h_q, _ = self.gru(zq, h_q_m1.permute(1, 0, 2))
        return((zp_mu, zp_std), (zq_mu, zq_std), h_q)

    def get_preds(self, action, z_mu, z_std, h_q_m1, quantity = 1):
        if(len(action.shape) == 2): action = action.unsqueeze(1)
        h_q_m1 = h_q_m1.permute(1, 0, 2)
        h, _ = self.gru(z_mu, h_q_m1)        
        
        rgbd = self.rgbd_up(torch.cat((h, action), dim=-1)).view((z_mu.shape[0], z_mu.shape[1], 4, self.args.image_size//2, self.args.image_size//2))
        rgbd_mu_pred = rnn_cnn(self.rgbd, rgbd).permute(0, 1, 3, 4, 2)
        spe_mu_pred  = self.spe(torch.cat((h, action), dim=-1))
                
        pred_rgbd = [] ; pred_spe = []
        for _ in range(quantity):
            z = sample(z_mu, z_std)
            h, _ = self.gru(z, h_q_m1)
            rgbd = self.rgbd_up(torch.cat((h, action), dim=-1)).view((z_mu.shape[0], z_mu.shape[1], 4, self.args.image_size//2, self.args.image_size//2))
            pred_rgbd.append((rnn_cnn(self.rgbd, rgbd).permute(0, 1, 3, 4, 2)))
            pred_spe.append(self.spe(torch.cat((h, action), dim=-1)))
        return((rgbd_mu_pred, pred_rgbd), (spe_mu_pred, pred_spe))



class Actor(nn.Module):

    def __init__(self, args = default_args):
        super(Actor, self).__init__()
        
        self.args = args
        rgbd_size = (1, 4, args.image_size, args.image_size)
        example = torch.zeros(rgbd_size)
        
        self.rgbd_in = nn.Sequential(
            ConstrainedConv2d(
                in_channels = 4,
                out_channels = 4,
                kernel_size = (3,3),
                padding = (1,1),
                padding_mode = "reflect"),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size = (3,3), 
                stride = (2,2),
                padding = (1,1)))
        example = self.rgbd_in(example).flatten(1)
        rgbd_size = example.shape[1]
        
        self.gru = nn.GRU(
            input_size =  rgbd_size + spe_size + action_size,
            hidden_size = args.hidden_size,
            batch_first = True)
        self.mu = nn.Sequential(
            nn.Linear(args.hidden_size, action_size))
        self.std = nn.Sequential(
            nn.Linear(args.hidden_size, action_size),
            nn.Softplus())

        self.rgbd_in.apply(init_weights)
        self.gru.apply(init_weights)
        self.mu.apply(init_weights)
        self.std.apply(init_weights)
        self.to(args.device)

    def forward(self, rgbd, spe, prev_action, h = None):
        if(len(rgbd.shape) == 4): rgbd = rgbd.unsqueeze(1)
        if(len(spe.shape) == 2):  spe =  spe.unsqueeze(1)
        rgbd = (rgbd * 2) - 1
        spe = (spe - self.args.min_speed) / (self.args.max_speed - self.args.min_speed)
        rgbd = rnn_cnn(self.rgbd_in, rgbd.permute(0, 1, 4, 2, 3)).flatten(2)
        h, _ = self.gru(torch.cat((rgbd, spe, prev_action), dim=-1), h)
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
        rgbd_size = (1, 4, args.image_size, args.image_size)
        example = torch.zeros(rgbd_size)
        
        self.rgbd_in = nn.Sequential(
            ConstrainedConv2d(
                in_channels = 4,
                out_channels = 4,
                kernel_size = (3,3),
                padding = (1,1),
                padding_mode = "reflect"),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size = (3,3), 
                stride = (2,2),
                padding = (1,1)))
        example = self.rgbd_in(example).flatten(1)
        rgbd_size = example.shape[1]
        
        self.gru = nn.GRU(
            input_size =  rgbd_size + spe_size + action_size,
            hidden_size = args.hidden_size,
            batch_first = True)
        self.lin = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, 1))

        self.rgbd_in.apply(init_weights)
        self.gru.apply(init_weights)
        self.lin.apply(init_weights)
        self.to(args.device)

    def forward(self, rgbd, spe, action, h = None):
        if(len(rgbd.shape) == 4): rgbd = rgbd.unsqueeze(1)
        if(len(spe.shape) == 2):  spe =  spe.unsqueeze(1)
        rgbd = (rgbd * 2) - 1
        spe = (spe - self.args.min_speed) / (self.args.max_speed - self.args.min_speed)
        rgbd = rnn_cnn(self.rgbd_in, rgbd.permute(0, 1, 4, 2, 3)).flatten(2)
        h, _ = self.gru(torch.cat((rgbd, spe, action), dim=-1), h)
        Q = self.lin(h)
        return(Q)
    
    
    
class Actor_HQ(nn.Module):

    def __init__(self, args = default_args):
        super(Actor_HQ, self).__init__()
        
        self.args = args
        
        self.lin = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.LeakyReLU(),
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
        return(action, log_prob, None)
    
    
    
class Critic_HQ(nn.Module):

    def __init__(self, args = default_args):
        super(Critic_HQ, self).__init__()
        
        self.args = args
        
        self.lin = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size + action_size, args.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.LeakyReLU(),
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
    print(torch_summary(forward, ((3, 1, args.image_size, args.image_size, 4), (3, 1, spe_size), (3, 1, action_size), (3, 1, args.hidden_size))))
    


    actor = Actor(args)
    
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor, ((3, 1, args.image_size, args.image_size, 4), (3, 1, spe_size), (3, 1, action_size))))
    
    
    
    critic = Critic(args)
    
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, ((3, 1, args.image_size, args.image_size, 4), (3, 1, spe_size), (3, 1, action_size))))
    
    
    
    actor = Actor_HQ(args)
    
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor, ((3, 1, args.hidden_size))))
    
    
    
    critic = Critic_HQ(args)
    
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, ((3, 1, args.hidden_size), (3, 1, action_size))))

# %%
