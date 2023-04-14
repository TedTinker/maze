#%% 

import torch
from torch import nn 
from torch.distributions import Normal
from torchinfo import summary as torch_summary

from utils import default_args, init_weights, ConstrainedConv2d
spe_size = 1 ; action_size = 2



def rnn_cnn(do_this, to_this):
    episodes = to_this.shape[0] ; steps = to_this.shape[1]
    this = to_this.view((episodes * steps, to_this.shape[2], to_this.shape[3], to_this.shape[4]))
    this = do_this(this)
    this = this.view((episodes, steps, this.shape[1], this.shape[2], this.shape[3]))
    return(this)

        

class State_Forward(nn.Module):
    
    def __init__(self, args = default_args):
        super(State_Forward, self).__init__()
        
        self.args = args
        rgbd_size = (1, 4, args.image_size, args.image_size)
        example = torch.zeros(rgbd_size)
        
        self.gru = nn.GRU(
            input_size =  args.hidden_size,
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
            nn.Linear(args.hidden_size + action_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.state_size))
        self.zp_rho = nn.Sequential(
            nn.Linear(args.hidden_size + action_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.state_size))
        
        self.zq_mu = nn.Sequential(
            nn.Linear(args.hidden_size + rgbd_size + spe_size + action_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.state_size))
        self.zq_rho = nn.Sequential(
            nn.Linear(args.hidden_size + rgbd_size + spe_size + action_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.state_size))
        
        self.rgbd_up = nn.Sequential(
            nn.Linear(args.hidden_size + action_size, rgbd_size),
            nn.LeakyReLU())
        self.rgbd_mu = nn.Sequential(
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
        self.rgbd_rho = nn.Sequential(
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
                kernel_size = (1,1)))
        
        self.spe_mu = nn.Sequential(
            nn.Linear(args.hidden_size + action_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, spe_size))
        self.spe_rho = nn.Sequential(
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
        self.rgbd_mu.apply(init_weights)
        self.rgbd_rho.apply(init_weights)
        self.spe_mu.apply(init_weights)
        self.spe_rho.apply(init_weights)
        self.to(args.device)
        
    def zp(self, prev_action, h = None):
        x = torch.cat((h, prev_action), dim=-1)
        zp_mu = self.zp_mu(x)
        zp_std = torch.log1p(torch.exp(self.zp_rho(x)))
        zp_std = torch.clamp(zp_std, min = self.args.std_min, max = self.args.std_max)
        return((zp_mu, zp_std))
        
    def zq(self, rgbd, spe, prev_action, h = None):
        rgbd = rnn_cnn(self.rgbd_in, rgbd.permute(0, 1, -1, 2, 3)).flatten(2)
        x = torch.cat((h, rgbd, spe, prev_action), dim=-1)
        zq_mu = self.zq_mu(x)
        zq_std = torch.log1p(torch.exp(self.zq_rho(x)))
        zq_std = torch.clamp(zq_std, min = self.args.std_min, max = self.args.std_max)
        e = Normal(0, 1).sample(zq_std.shape).to("cuda" if next(self.parameters()).is_cuda else "cpu")
        zq = zq_mu + e * zq_std
        return(zq, (zq_mu, zq_std))
        
    def forward(self, rgbd, spe, prev_action, action, h = None):
        if(len(rgbd.shape) == 4): rgbd = rgbd.unsqueeze(1)
        if(len(spe.shape) == 2): spe = spe.unsqueeze(1)
        if(len(prev_action.shape) == 2): prev_action = prev_action.unsqueeze(1)
        rgbd = (rgbd * 2) - 1
        spe = (spe - self.args.min_speed) / (self.args.max_speed - self.args.min_speed)
        if(h == None): h = torch.zeros((spe.shape[0], 1, self.args.hidden_size)).to(spe.device)
        zp_dist = self.zp(prev_action, h)
        zq, zq_dist = self.zq(rgbd, spe, prev_action, h)
        h = h if h == None else h.permute(1, 0, 2)
        h, _ = self.gru(zq, h)
        x = torch.cat((h, action.unsqueeze(1)), dim=-1)
        
        rgbd_x = self.rgbd_up(x).view((spe.shape[0], spe.shape[1], 4, 4, 4))
        rgbd_mu = (rnn_cnn(self.rgbd_mu, rgbd_x) + 1) / 2
        rgbd_mu = rgbd_mu.permute(0, 1, 3, 4, 2)
        rgbd_std = torch.log1p(torch.exp(rnn_cnn(self.rgbd_rho, rgbd_x)))
        rgbd_std = torch.clamp(rgbd_std, min = self.args.std_min, max = self.args.std_max)
        rgbd_std = rgbd_std.permute(0, 1, 3, 4, 2)
        e = Normal(0, 1).sample(rgbd_std.shape).to("cuda" if next(self.parameters()).is_cuda else "cpu")
        pred_rgbd = rgbd_mu + e * rgbd_std
        
        spe_mu = self.spe_mu(x)
        spe_std = torch.log1p(torch.exp(self.spe_rho(x)))
        spe_std = torch.clamp(spe_std, min = self.args.std_min, max = self.args.std_max)
        e = Normal(0, 1).sample(spe_std.shape).to("cuda" if next(self.parameters()).is_cuda else "cpu")
        pred_spe = spe_mu + e * spe_std
        return(pred_rgbd, (rgbd_mu, rgbd_std), pred_spe, (spe_mu, spe_std), zp_dist, zq_dist, h)



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
        
        self.gru = nn.GRU(
            input_size =  rgbd_size + spe_size + action_size,
            hidden_size = args.hidden_size,
            batch_first = True)

        self.rgbd_up = nn.Sequential(
            nn.Linear(args.hidden_size + action_size, rgbd_size),
            nn.LeakyReLU())
        self.rgbd_mu = nn.Sequential(
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
        self.rgbd_rho = nn.Sequential(
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
                kernel_size = (1,1)))
        
        self.spe_mu = nn.Sequential(
            nn.Linear(args.hidden_size + action_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, spe_size))
        self.spe_rho = nn.Sequential(
            nn.Linear(args.hidden_size + action_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.hidden_size), 
            nn.Tanh(),
            nn.Linear(args.hidden_size, spe_size))
        
        self.rgbd_in.apply(init_weights)
        self.gru.apply(init_weights)
        self.rgbd_up.apply(init_weights)
        self.rgbd_mu.apply(init_weights)
        self.rgbd_rho.apply(init_weights)
        self.spe_mu.apply(init_weights)
        self.spe_rho.apply(init_weights)
        self.to(args.device)
        
    def forward(self, rgbd, spe, prev_action, action, h = None):
        if(len(rgbd.shape) == 4): rgbd = rgbd.unsqueeze(1)
        if(len(spe.shape) == 2): spe = spe.unsqueeze(1)
        if(len(prev_action.shape) == 2): prev_action = prev_action.unsqueeze(1)
        rgbd = (rgbd * 2) - 1
        spe = (spe - self.args.min_speed) / (self.args.max_speed - self.args.min_speed)
        rgbd = rnn_cnn(self.rgbd_in, rgbd.permute(0, 1, -1, 2, 3)).flatten(2)
        x = torch.cat((rgbd, spe, prev_action), dim=-1)
        h, _ = self.gru(x, h)
        x = torch.cat((h, action), dim=-1)

        rgbd_x = self.rgbd_up(x).view((spe.shape[0], spe.shape[1], 4, 4, 4))
        rgbd_mu = (rnn_cnn(self.rgbd_mu, rgbd_x) + 1) / 2
        rgbd_mu = rgbd_mu.permute(0, 1, 3, 4, 2)
        rgbd_std = torch.log1p(torch.exp(rnn_cnn(self.rgbd_rho, rgbd_x)))
        rgbd_std = torch.clamp(rgbd_std, min = self.args.std_min, max = self.args.std_max)
        rgbd_std = rgbd_std.permute(0, 1, 3, 4, 2)
        e = Normal(0, 1).sample(rgbd_std.shape).to("cuda" if next(self.parameters()).is_cuda else "cpu")
        pred_rgbd = rgbd_mu + e * rgbd_std
        
        spe_mu = self.spe_mu(x)
        spe_std = torch.log1p(torch.exp(self.spe_rho(x)))
        spe_std = torch.clamp(spe_std, min = self.args.std_min, max = self.args.std_max)
        e = Normal(0, 1).sample(spe_std.shape).to("cuda" if next(self.parameters()).is_cuda else "cpu")
        pred_spe = spe_mu + e * spe_std

        return(pred_rgbd, (rgbd_mu, rgbd_std), pred_spe, (spe_mu, spe_std))
        


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
        if(len(spe.shape) == 2): spe = spe.unsqueeze(1)
        if(len(prev_action.shape) == 2): prev_action = prev_action.unsqueeze(1)
        rgbd = (rgbd * 2) - 1
        spe = (spe - self.args.min_speed) / (self.args.max_speed - self.args.min_speed)
        rgbd = rnn_cnn(self.rgbd_in, rgbd.permute(0, 1, -1, 2, 3)).flatten(2)
        x = torch.cat((rgbd, spe, prev_action), dim=-1)
        h, _ = self.gru(x, h)
        mu = self.mu(h)
        std = torch.log1p(torch.exp(self.rho(h)))
        std = torch.clamp(std, min = self.args.std_min, max = self.args.std_max)
        e = Normal(0, 1).sample(std.shape).to("cuda" if next(self.parameters()).is_cuda else "cpu")
        x = mu + e * std
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
            nn.Linear(args.hidden_size + action_size, args.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, 1))

        self.rgbd_in.apply(init_weights)
        self.gru.apply(init_weights)
        self.lin.apply(init_weights)
        self.to(args.device)

    def forward(self, rgbd, spe, prev_action, action, h = None):
        if(len(rgbd.shape) == 4): rgbd = rgbd.unsqueeze(1)
        if(len(spe.shape) == 2): spe = spe.unsqueeze(1)
        if(len(prev_action.shape) == 2): prev_action = prev_action.unsqueeze(1)
        rgbd = (rgbd * 2) - 1
        spe = (spe - self.args.min_speed) / (self.args.max_speed - self.args.min_speed)
        rgbd = rnn_cnn(self.rgbd_in, rgbd.permute(0, 1, -1, 2, 3)).flatten(2)
        x = torch.cat((rgbd, spe, prev_action), dim=-1)
        h, _ = self.gru(x, h)
        x = torch.cat((h, action), dim=-1)
        x = self.lin(x)
        return(x)
    


if __name__ == "__main__":
    
    args = default_args
    args.device = "cpu"
    args.dkl_rate = 1
    
    
    
    forward = State_Forward(args)
    
    print("\n\n")
    print(forward)
    print()
    print(torch_summary(forward, ((3, args.image_size, args.image_size, 4), (3, spe_size), (3, action_size), (3, action_size))))
    
    
    
    forward = Forward(args)
    
    print("\n\n")
    print(forward)
    print()
    print(torch_summary(forward, ((3, 1, args.image_size, args.image_size, 4), (3, 1, spe_size), (3, 1, action_size), (3, 1, action_size))))
    


    actor = Actor(args)
    
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor, ((3, 1, args.image_size, args.image_size, 4), (3, 1, spe_size), (3, 1, action_size))))
    
    
    
    critic = Critic(args)
    
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, ((3, 1, args.image_size, args.image_size, 4), (3, 1, spe_size), (3, 1, action_size), (3, 1, action_size))))

# %%
