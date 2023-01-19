#%% 

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass
    
    
    
import sys ; sys.argv=[''] ; del sys
import argparse

parser = argparse.ArgumentParser()

# Training 
parser.add_argument('--max_steps',          type=int,   default = 10)
parser.add_argument('--epochs',             type=int,   default = 1000)
parser.add_argument('--batch_size',         type=int,   default = 16)
parser.add_argument('--GAMMA',              type=int,   default = .99)

# Module 
parser.add_argument('--bayes',              type=bool,  default = False)
parser.add_argument('--forward_lr',         type=float, default = .01)
parser.add_argument('--actor_lr',           type=float, default = .01) 
parser.add_argument('--critic_lr',          type=float, default = .01) 
parser.add_argument('--alpha_lr',           type=float, default = .01) 

# Memory buffer
parser.add_argument('--capacity',           type=int,   default = 100)
parser.add_argument('--replacement',        type=str,   default = "index")
parser.add_argument('--selection',          type=str,   default = "uniform")
parser.add_argument('--power',              type=float, default = 1)

# Training
parser.add_argument("--d",                  type=int,   default = 2)    # Delay to train actors
parser.add_argument("--alpha",              type=float, default = 0) # Soft-Actor-Critic entropy aim
parser.add_argument("--target_entropy",     type=float, default = -2)   # Soft-Actor-Critic entropy aim
parser.add_argument("--eta",                type=float, default = 0) # Scale curiosity
parser.add_argument("--tau",                type=float, default = .05)  # For soft-updating target critics
parser.add_argument("--dkl_rate",           type=float, default = .001)#.0001)# Scale bayesian dkl
parser.add_argument("--sample_elbo",        type=int,   default = 5)   # Samples for elbo
parser.add_argument("--naive_curiosity",    type=str,   default = True) # Which kind of curiosity
parser.add_argument("--dkl_change_size",    type=str,   default = "batch")  # "batch", "episode", "step"

args, _ = parser.parse_known_args()
    
    

from random import choice

class Spot:
    
    def __init__(self, pos, exit_reward = None, name = "NONE"):
        self.pos = pos ; self.exit_reward = exit_reward
        self.name = name

class T_Maze:
    
    def __init__(self):
        self.maze = [
            Spot((0, 0)), Spot((0, 1)), 
            Spot((-1, 1), 1, "BAD"), Spot((1, 1)), Spot((1, 2)), 
            Spot((2, 2)), Spot((3, 2)), Spot((3, 1), 10, "GOOD")]
        self.agent_pos = (0, 0)
        
    def obs(self):
        right = 0 ; left = 0 ; up = 0 ; down = 0 
        for spot in self.maze:
            if(spot.pos == (self.agent_pos[0]+1, self.agent_pos[1])): right = 1 
            if(spot.pos == (self.agent_pos[0]-1, self.agent_pos[1])): left = 1 
            if(spot.pos == (self.agent_pos[0], self.agent_pos[1]+1)): up = 1 
            if(spot.pos == (self.agent_pos[0], self.agent_pos[1]-1)): down = 1 
        return(torch.tensor((self.agent_pos[0], self.agent_pos[1], right, left, up, down)).unsqueeze(0).float())
        
    def action(self, x = 0, y = 0):
        if(abs(x) > abs(y)):
            if(x < 0): x = -1 ; y = 0 
            else:      x = 1  ; y = 0
        else:
            if(y < 0): x = 0  ; y = -1 
            else:      x = 0  ; y = 1
        new_pos = (self.agent_pos[0] + x, self.agent_pos[1] + y)
        for spot in self.maze:
            if(spot.pos == new_pos):
                self.agent_pos = new_pos 
                if(spot.exit_reward == None):
                    return(0, spot.name, False)
                else:
                    if(type(spot.exit_reward) == tuple):
                        return(choice(spot.exit_reward), spot.name, True)
                    else:
                        return(spot.exit_reward, spot.name, True)
                break
        return(-1, "NONE", False)    
    
    def __str__(self):
        to_print = ""
        for y in [2, 1, 0]:
            for x in [-1, 0, 1, 2, 3]:
                portrayal = " "
                for spot in self.maze:
                    if(spot.pos == (x, y)): portrayal = "O"
                if(self.agent_pos == (x, y)): portrayal = "X"
                to_print += portrayal 
            to_print += "\n"
        return(to_print)
    
    
    
if __name__ == "__main__":
    t_maze = T_Maze()
    print(t_maze)
    print(t_maze.obs())
    
    reward, name, done = t_maze.action(1, 0)
    print(t_maze)
    print(reward, name, done, "\n")
    print(t_maze.obs())
    
    reward, name, done = t_maze.action(0, 1)
    print(t_maze)
    print(reward, name, done, "\n")
    print(t_maze.obs())
    
    reward, name, done = t_maze.action(-1, 0)
    print(t_maze)
    print(reward, name, done, "\n")
    print(t_maze.obs())
    
    

from blitz.modules.base_bayesian_module import BayesianModule

def weights(model):
    weight_mu = [] ; weight_sigma = []
    bias_mu = [] ;   bias_sigma = []
    for module in model.modules():
        if isinstance(module, (BayesianModule)):
            weight_mu.append(module.weight_sampler.mu.clone().flatten())
            weight_sigma.append(torch.log1p(torch.exp(module.weight_sampler.rho.clone().flatten())))
            bias_mu.append(module.bias_sampler.mu.clone().flatten()) 
            bias_sigma.append(torch.log1p(torch.exp(module.bias_sampler.rho.clone().flatten())))
    if(weight_mu == []):
        return(torch.zeros([1]), torch.zeros([1]), torch.zeros([1]), torch.zeros([1]))
    return(
        torch.cat(weight_mu, -1).to("cpu"),
        torch.cat(weight_sigma, -1).to("cpu"),
        torch.cat(bias_mu, -1).to("cpu"),
        torch.cat(bias_sigma, -1).to("cpu"))
    
def dkl(mu_1, sigma_1, mu_2, sigma_2):
    sigma_1 = torch.pow(sigma_1, 2)
    sigma_2 = torch.pow(sigma_2, 2)
    term_1 = torch.pow(mu_2 - mu_1, 2) / sigma_2 
    term_2 = sigma_1 / sigma_2 
    term_3 = torch.log(term_2)
    out = (.5 * (term_1 + term_2 - term_3 - 1)).sum()
    out = torch.nan_to_num(out)
    return(out)


import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal

import matplotlib.pyplot as plt 
from itertools import accumulate
import numpy as np

def plot_rewards(rewards, e):
    rewards = list(accumulate(rewards))
    plt.plot(rewards) 
    plt.title("Cumulative Rewards at epoch {}".format(e))
    plt.show()
    plt.close()

def plot_spot_names(spot_names, e):
    kinds = ["NONE", "BAD", "GOOD"]
    plt.scatter([0 for _ in kinds], kinds, color = (0,0,0,0))
    plt.scatter(range(len(spot_names)), spot_names, color = "gray")
    plt.title("Endings at epoch {}".format(e))
    plt.show()
    plt.close()
    
def plot_losses(losses, e):
    losses = np.concatenate(losses, axis = 0)
    plt.plot(losses[:,0], color = "blue", label = "MSE")
    plt.plot(losses[:,1], color = "red", label = "DKL")
    plt.legend()
    plt.title("Forward Losses at epoch {}".format(e))
    plt.show()
    plt.close()
    
    alpha = losses[:,2]
    alpha_xs = [i for i in range(len(alpha)) if alpha[i] != None] ; alpha = [a for a in alpha if a != None]
    
    actor = losses[:,3]
    actor_xs = [i for i in range(len(actor)) if actor[i] != None] ; actor = [a for a in actor if a != None]
    
    plt.plot(alpha_xs, alpha, color = "black", label = "Alpha", alpha = .5)
    ax2 = plt.twinx()
    ax2.plot(actor_xs, actor, color = "red", label = "Actor", alpha = .5)
    ax3 = plt.twinx()
    ax3.plot(losses[:,4], color = "blue", label = "Critic 1", alpha = .5)
    ax3.plot(losses[:,5], color = "blue", label = "Critic 2", alpha = .5)
    plt.legend()
    plt.title("Other Losses at epoch {}".format(e))
    plt.show()
    plt.close()
    
def plot_ext_int(extrinsic, intrinsic_curiosity, intrinsic_entropy, e):
    ent_xs = [i for i in range(len(intrinsic_entropy)) if intrinsic_entropy[i] != None] ; intrinsic_entropy = [a for a in intrinsic_entropy if a != None]

    plt.plot(extrinsic, color = "red", label = "Extrinsic", alpha = .5)
    plt.plot(intrinsic_curiosity, color = "green", label = "Curiosity", alpha = .5)
    plt.plot(ent_xs, intrinsic_entropy, color = "blue", label = "Entropy", alpha = .5)
    plt.legend()
    plt.title("Extrinsic and Intrinsic Rewards")
    plt.show()
    plt.close()
    
def plot_dkl_change(dkl_change, e):
    plt.plot(dkl_change) 
    plt.title("DKL at epoch {}".format(e))
    plt.show()
    plt.close()
                
            
            

    
# %%
