#%% 

import torch
from torch import nn 
from torch.distributions import Normal
from torchinfo import summary as torch_summary

from utils import default_args, init_weights, ConstrainedConv2d
spe_size = 1 ; action_size = 2



def var(x, mu_func, rho_func, args):
    mu = mu_func(x)
    std = torch.log1p(torch.exp(rho_func(x)))
    std = torch.clamp(std, min = args.std_min, max = args.std_max)
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
        
        self.gru = nn.GRU(
            input_size =  args.state_size,
            hidden_size = args.hidden_size,
            batch_first = True)
        
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
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.state_size),
            nn.Tanh())
        self.zp_rho = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.state_size))
        
        self.zq_mu = nn.Sequential(
            nn.Linear(args.hidden_size + rgbd_size + spe_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.state_size),
            nn.Tanh())
        self.zq_rho = nn.Sequential(
            nn.Linear(args.hidden_size + rgbd_size + spe_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.state_size))
        
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
            nn.Tanh(),
            nn.Upsample(
                scale_factor = 2, 
                mode = "bilinear",
                align_corners = True),
            ConstrainedConv2d(
                in_channels = 4,
                out_channels = 4,
                kernel_size = (1,1)),
            nn.Tanh())
        
        self.spe = nn.Sequential(
            nn.Linear(args.hidden_size + action_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, spe_size))
        
        self.gru.apply(init_weights)
        self.rgbd_in.apply(init_weights)
        self.zp_mu.apply(init_weights)
        self.zp_rho.apply(init_weights)
        self.zq_mu.apply(init_weights)
        self.zq_rho.apply(init_weights)
        self.rgbd_up.apply(init_weights)
        self.rgbd.apply(init_weights)
        self.spe.apply(init_weights)
        self.to(args.device)
        
    def forward(self, rgbd, spe, h_q_m1):
        if(len(rgbd.shape) == 4): rgbd = rgbd.unsqueeze(1)
        if(len(spe.shape) == 2):  spe =  spe.unsqueeze(1)
        rgbd = (rgbd * 2) - 1
        spe = (spe - self.args.min_speed) / (self.args.max_speed - self.args.min_speed)
        rgbd = rnn_cnn(self.rgbd_in, rgbd.permute(0, 1, -1, 2, 3)).flatten(2)
        zp_mu, zp_std = var(h_q_m1, self.zp_mu, self.zp_rho, self.args)
        zq_mu, zq_std = var(torch.cat((h_q_m1, rgbd, spe), dim=-1), self.zq_mu, self.zq_rho, self.args)        
        zq = sample(zq_mu, zq_std)
        h_q, _ = self.gru(zq, h_q_m1.permute(1, 0, 2))
        return((zp_mu, zp_std), (zq_mu, zq_std), h_q)

    def get_preds(self, action, z_mu, z_std, h_q_m1, quantity = 1):
        if(len(action.shape) == 2): action = action.unsqueeze(1)
        h_q_m1 = h_q_m1.permute(1, 0, 2)
        h, _ = self.gru(z_mu, h_q_m1)        
        
        rgbd = self.rgbd_up(torch.cat((h, action), dim=-1)).view((action.shape[0], action.shape[1], self.args.image_size//2, self.args.image_size//2, 4))
        rgbd_mu_pred = rnn_cnn(self.rgbd, rgbd)
        spe_mu_pred  = self.spe(torch.cat((h, action), dim=-1))
        pred_rgbd = [] ; pred_spe = []
        for _ in range(quantity):
            z = sample(z_mu, z_std)
            h, _ = self.gru(z, h_q_m1)
            rgbd = self.rgbd_up(torch.cat((h, action), dim=-1)).view((action.shape[0], action.shape[1], self.args.image_size//2, self.args.image_size//2, 4))
            pred_rgbd.append(rnn_cnn(self.rgbd, rgbd))
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
        self.rho = nn.Sequential(
            nn.Linear(args.hidden_size, action_size))

        self.rgbd_in.apply(init_weights)
        self.gru.apply(init_weights)
        self.mu.apply(init_weights)
        self.rho.apply(init_weights)
        self.to(args.device)

    def forward(self, rgbd, spe, prev_action, h = None):
        if(len(rgbd.shape) == 4): rgbd = rgbd.unsqueeze(1)
        if(len(spe.shape) == 2):  spe =  spe.unsqueeze(1)
        rgbd = (rgbd * 2) - 1
        spe = (spe - self.args.min_speed) / (self.args.max_speed - self.args.min_speed)
        rgbd = rnn_cnn(self.rgbd_in, rgbd.permute(0, 1, -1, 2, 3)).flatten(2)
        x = torch.cat((rgbd, spe, prev_action), dim=-1)
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
        rgbd = rnn_cnn(self.rgbd_in, rgbd.permute(0, 1, -1, 2, 3)).flatten(2)
        x = torch.cat((rgbd, spe, action), dim=-1)
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
    print(torch_summary(forward, ((3, 1, args.image_size, args.image_size, 4), (3, 1, spe_size), (3, 1, args.hidden_size))))
    


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

# %%
