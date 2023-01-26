#%% 

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys ; sys.argv=[''] ; del sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--device',             type=str,   default = "cpu")

# Training 
parser.add_argument('--agents',             type=int,   default = 10)
parser.add_argument('--max_steps',          type=int,   default = 10)
parser.add_argument('--epochs',             type=int,   default = 1000)
parser.add_argument('--batch_size',         type=int,   default = 8)
parser.add_argument('--GAMMA',              type=int,   default = .99)

# Module 
parser.add_argument('--hidden',             type=int,   default = 32)
parser.add_argument('--dkl_hidden',         type=int,   default = 4)
parser.add_argument('--bias',               type=bool,  default = True)
parser.add_argument('--forward_lr',         type=float, default = .01)
parser.add_argument('--clone_lr',           type=float, default = .001)
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
parser.add_argument("--alpha",              type=float, default = 0)    # Soft-Actor-Critic entropy aim
parser.add_argument("--target_entropy",     type=float, default = -2)   # Soft-Actor-Critic entropy aim
parser.add_argument("--naive_eta",          type=float, default = 1)    # Scale curiosity
parser.add_argument("--friston_eta",        type=float, default = .0001)# Scale curiosity
parser.add_argument("--tau",                type=float, default = .05)  # For soft-updating target critics
parser.add_argument("--dkl_rate",           type=float, default = .001) # Scale bayesian dkl
parser.add_argument("--sample_elbo",        type=int,   default = 5)    # Samples for elbo
parser.add_argument("--curiosity",          type=str,   default = "none") # Which kind of curiosity
parser.add_argument("--dkl_change_size",    type=str,   default = "batch")  # "batch", "step"

default_args, _ = parser.parse_known_args()



def get_title(arg_dict):
    parser = argparse.ArgumentParser()
    for arg in vars(default_args):
        if(arg in arg_dict.keys()): parser.add_argument('--{}'.format(arg), default = arg_dict[arg])
        else:                       parser.add_argument('--{}'.format(arg), default = getattr(default_args, arg))
    args, _ = parser.parse_known_args()
    title = ""
    first = True
    for arg in vars(args):
        if(getattr(args, arg) != getattr(default_args, arg)):
            if(not first): title += "_"
            title += "{}_{}".format(arg, getattr(args, arg)) ; first = False
    if(len(title) == 0): title = "default"
    print(arg_dict, title)
    return(args, title)



def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass
    
    

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



if __name__ == "__main__":

    mu_1 = torch.tensor([0])
    mu_2 = torch.tensor([10])
    sigma_1 = torch.tensor([2])
    sigma_2 = torch.tensor([30])

    print(dkl(mu_1, sigma_1, mu_2, sigma_2))


# %%
