#%% 

import argparse
#import sys ; sys.argv=[''] ; del sys
import os 

if(os.getcwd().split("/")[-1] != "easy_maze"): os.chdir("easy_maze")
print(os.getcwd())

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

# Meta 
parser.add_argument("--arg_title",          type=str,   default = "default") 
parser.add_argument("--name",               type=str,   default = "default") 
parser.add_argument("--id",                 type=int,   default = 0)
parser.add_argument('--device',             type=str,   default = "cpu")

# Maze 
parser.add_argument('--max_steps',          type=int,   default = 10)
parser.add_argument('--step_lim_punishment',type=int,   default = -1)
parser.add_argument('--wall_punishment',    type=int,   default = -1)

# Module 
parser.add_argument('--hidden',             type=int,   default = 32)
parser.add_argument("--complexity",         type=float, default = .005)   # Scale complexity loss
parser.add_argument("--sample_elbo",        type=int,   default = 10)  # Samples for elbo
parser.add_argument('--forward_lr',         type=float, default = .01)
parser.add_argument('--alpha_lr',           type=float, default = .01) 
parser.add_argument('--actor_lr',           type=float, default = .01)
parser.add_argument('--critic_lr',          type=float, default = .01)

# Memory buffer
parser.add_argument('--capacity',           type=int,   default = 100)

# Training
parser.add_argument('--epochs',             type=int,   default = 1000)
parser.add_argument('--batch_size',         type=int,   default = 8)
parser.add_argument('--GAMMA',              type=int,   default = .99)
parser.add_argument("--d",                  type=int,   default = 2)        # Delay to train actors
parser.add_argument("--alpha",              type=str,   default = 0)        # Soft-Actor-Critic entropy aim
parser.add_argument("--target_entropy",     type=float, default = -2)       # Soft-Actor-Critic entropy aim
parser.add_argument("--naive_eta",          type=float, default = 1)        # Scale curiosity
parser.add_argument("--free_eta",           type=float, default = 5)        # Scale curiosity
parser.add_argument("--tau",                type=float, default = .05)      # For soft-updating target critics
parser.add_argument("--curiosity",          type=str,   default = "none")     # Which kind of curiosity

# Saving data
parser.add_argument('--keep_data',          type=int,   default = 10)



try:
    default_args = parser.parse_args([])
    try:    args    = parser.parse_args()
    except: args, _ = parser.parse_known_args()
except:
    import sys ; sys.argv=[''] ; del sys           # Comment this out when using bash
    default_args = parser.parse_args([])
    try:    args    = parser.parse_args()
    except: args, _ = parser.parse_known_args()

if(default_args.alpha == "None"): default_args.alpha = None
if(args.alpha == "None"):         args.alpha = None

for arg in vars(default_args):
    if(getattr(default_args, arg) == "None"):  default_args.arg = None
    if(getattr(default_args, arg) == "True"):  default_args.arg = True
    if(getattr(default_args, arg) == "False"): default_args.arg = False
    if(getattr(args, arg) == "None"):  args.arg = None
    if(getattr(args, arg) == "True"):  args.arg = True
    if(getattr(args, arg) == "False"): args.arg = False



def get_args_name(default_args, args):
    name = "" ; first = True
    for arg in vars(default_args):
        if(arg in ["arg_title", "id"]): pass 
        else: 
            default, this_time = getattr(default_args, arg), getattr(args, arg)
            if(this_time == default): pass
            else: 
                if first: first = False
                else: name += "_"
                name += "{}_{}".format(arg, this_time)
    if(name == ""): name = "default" 
    return(name)



if(args.name[:3] != "___"):
    name = get_args_name(default_args, args)
    args.name = name

folder = "saved/" + args.arg_title
if(args.arg_title[:3] != "___" and args.arg_title != "default"):
    try: os.mkdir(folder)
    except: pass
if(default_args.alpha == "None"): default_args.alpha = None
if(args.alpha == "None"):         args.alpha = None

if(args == default_args): print("Using default arguments.")
else:
    for arg in vars(default_args):
        default, this_time = getattr(default_args, arg), getattr(args, arg)
        if(this_time == default): pass
        else: print("{}:\n\tDefault:\t{}\n\tThis time:\t{}".format(arg, default, this_time))
print("\n\n")



def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass


    
def dkl(mu_1, std_1, mu_2, std_2):
    std_1 = torch.pow(std_1, 2)
    std_2 = torch.pow(std_2, 2)
    term_1 = torch.pow(mu_2 - mu_1, 2) / std_2 
    term_2 = std_1 / std_2 
    term_3 = torch.log(term_2)
    out = (.5 * (term_1 + term_2 - term_3 - 1))
    out = torch.nan_to_num(out)
    return(out)