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
parser.add_argument('--capacity',           type=int,   default = 300)
parser.add_argument('--replacement',        type=str,   default = "index")
parser.add_argument('--selection',          type=str,   default = "uniform")
parser.add_argument('--power',              type=float, default = 1)

# Training
parser.add_argument("--d",                  type=int,   default = 2)    # Delay to train actors
parser.add_argument("--alpha",              type=float, default = None) # Soft-Actor-Critic entropy aim
parser.add_argument("--target_entropy",     type=float, default = -5)   # Soft-Actor-Critic entropy aim
parser.add_argument("--eta",                type=float, default = 0) # Scale curiosity
parser.add_argument("--eta_rate",           type=float, default = 1)    # Scale eta
parser.add_argument("--tau",                type=float, default = .05)  # For soft-updating target critics
parser.add_argument("--dkl_rate",           type=float, default = .001)#.0001)# Scale bayesian dkl
parser.add_argument("--sample_elbo",        type=int,   default = 5)   # Samples for elbo
parser.add_argument("--naive_curiosity",    type=str,   default = "true") # Which kind of curiosity
parser.add_argument("--dkl_change_size",    type=str,   default = "batch")  # "batch", "episode", "step"

args, _ = parser.parse_known_args()
    
    

from random import choice
from itertools import product

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
        
    def action(self, x = 0, y = 0):
        if(abs(x) > abs(y)):
            if(x < 0): x = -1 ; y = 0 
            else:      x = 1  ; y = 0
        else:
            if(y < 0): x = 0 ; y = -1 
            else:      x = 0 ; y = 1
        new_pos = (self.agent_pos[0] + x, self.agent_pos[1] + y)
        for spot in self.maze:
            if(spot.pos == new_pos):
                self.agent_pos = new_pos 
                if(spot.exit_reward != None):
                    if(type(spot.exit_reward) == tuple):
                        return(choice(spot.exit_reward), spot.name)
                    else:
                        return(spot.exit_reward, spot.name)
                break
        return(0, "NONE")    
    
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
    return((.5 * (term_1 + term_2 - term_3 - 1)).sum())



import matplotlib.pyplot as plt 
from itertools import accumulate

def plot_rewards(rewards, e):
    rewards = list(accumulate(rewards))
    plt.plot(rewards) 
    plt.title("Rewards at epoch {}".format(e))
    plt.show()
    plt.close()

def plot_spot_names(spot_names, e):
    kinds = ["NONE", "BAD", "GOOD"]
    plt.scatter([0 for _ in kinds], kinds, color = (0,0,0,0))
    plt.scatter(range(len(spot_names)), spot_names, color = "gray")
    plt.title("Endings at epoch {}".format(e))
    plt.show()
    plt.close()
                
            
            
if __name__ == "__main__":
    t_maze = T_Maze()
    print(t_maze)
    t_maze.action(1, 0)
    print(t_maze)
    t_maze.action(0, 1)
    print(t_maze)
    
# %%
