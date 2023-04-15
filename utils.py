#%% 

import datetime 

start_time = datetime.datetime.now()
    
def duration(start_time = start_time):
    change_time = datetime.datetime.now() - start_time
    change_time = change_time - datetime.timedelta(microseconds=change_time.microseconds)
    return(change_time)

def estimate_total_duration(proportion_completed, start_time=start_time):
    if(proportion_completed != 0): 
        so_far = datetime.datetime.now() - start_time
        estimated_total = so_far / proportion_completed
        estimated_total = estimated_total - datetime.timedelta(microseconds=estimated_total.microseconds)
    else: estimated_total = "?:??:??"
    return(estimated_total)



import argparse, ast, os
from math import exp, pi

if(os.getcwd().split("/")[-1] != "maze"): os.chdir("maze")

import torch
from torch import nn 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

def tuple_type(arg_string): return(ast.literal_eval(arg_string))

# Meta 
parser.add_argument("--arg_title",          type=str,   default = "default") 
parser.add_argument("--arg_name",           type=str,   default = "default") 
parser.add_argument("--agents",             type=int,   default = 36)
parser.add_argument("--previous_agents",    type=int,   default = 0)
parser.add_argument('--device',             type=str,   default = "cpu")

# Maze 
parser.add_argument('--hard_maze',          type=bool,  default = False)
parser.add_argument('--maze_list',          type=tuple_type,   default = ("easy",))
parser.add_argument('--max_steps',          type=int,   default = 10)
parser.add_argument('--step_lim_punishment',type=int,   default = -1)
parser.add_argument('--wall_punishment',    type=int,   default = -1)
parser.add_argument('--non_one',            type=int,   default = -1)
parser.add_argument('--default_reward',     type=tuple_type, default = ((1, 1),))  # ((weight, reward), (weight, reward))
parser.add_argument('--better_reward',      type=tuple_type, default = ((1, 10),))
parser.add_argument('--randomness',         type=bool,  default = 0)

# Hard Maze
parser.add_argument('--body_size',          type=float, default = 2)    
parser.add_argument('--image_size',         type=int,   default = 8)
parser.add_argument('--min_speed',          type=float, default = 50)
parser.add_argument('--max_speed',          type=float, default = 100)
parser.add_argument('--steps_per_step',     type=int,   default = 5)
parser.add_argument('--max_yaw_change',     type=float, default = pi/2)

# Module 
parser.add_argument('--hidden_size',        type=int,   default = 32)   
parser.add_argument('--std_min',            type=int,   default = exp(-20))
parser.add_argument('--std_max',            type=int,   default = exp(2))
parser.add_argument('--state_size',         type=int,   default = 32)
parser.add_argument("--beta_obs",           type=float, default = 2)
parser.add_argument("--beta_zq",            type=float, default = 0)      
parser.add_argument("--sigma_obs",          type=float, default = 1)     
parser.add_argument("--sigma_zq",           type=float, default = 1)      
parser.add_argument('--state_forward',      type=str,   default = False)
parser.add_argument('--forward_lr',         type=float, default = .01)
parser.add_argument('--alpha_lr',           type=float, default = .01) 
parser.add_argument('--actor_lr',           type=float, default = .01)
parser.add_argument('--critic_lr',          type=float, default = .01)
parser.add_argument('--action_prior',       type=str,   default = "normal")

# Memory buffer
parser.add_argument('--capacity',           type=int,   default = 100)

# Training
parser.add_argument('--epochs',             type=tuple_type,   default = (1000,))
parser.add_argument('--steps_per_epoch',    type=int,   default = 10)
parser.add_argument('--batch_size',         type=int,   default = 8)
parser.add_argument('--GAMMA',              type=int,   default = .99)
parser.add_argument("--d",                  type=int,   default = 2)        # Delay to train actors
parser.add_argument("--alpha",              type=str,   default = 0)        # Soft-Actor-Critic entropy aim
parser.add_argument("--target_entropy",     type=float, default = -2)       # Soft-Actor-Critic entropy aim
parser.add_argument("--naive_eta",          type=float, default = .25)        # Scale curiosity
parser.add_argument("--free_eta_obs",       type=float, default = 10)        # Scale curiosity
parser.add_argument("--free_eta_zq",        type=float, default = 0)        # Scale curiosity
parser.add_argument("--tau",                type=float, default = .05)      # For soft-updating target critics
parser.add_argument('--accuracy',           type=str,   default = "mse")
parser.add_argument("--curiosity",          type=str,   default = "none")     # Which kind of curiosity

# Saving data
parser.add_argument('--keep_data',           type=int,   default = 25)
parser.add_argument('--epochs_per_pos_list', type=int,   default = 250)
parser.add_argument('--episodes_in_pos_list',type=int,   default = 3)
parser.add_argument('--agents_per_pos_list', type=int,   default = -1)




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



args_not_in_title = ["arg_title", "id", "agents", "previous_agents", "hard_maze", "maze_list", "keep_data", "epochs_per_pos_list", "episodes_in_pos_list", "agents_per_pos_list"]
def get_args_title(default_args, args):
    if(args.arg_name[:3] == "___"): return("plotting")
    name = "" ; first = True
    arg_list = list(vars(default_args).keys())
    arg_list.insert(0, arg_list.pop(arg_list.index("arg_name")))
    for arg in arg_list:
        if(arg in args_not_in_title): pass 
        else: 
            default, this_time = getattr(default_args, arg), getattr(args, arg)
            if(this_time == default): pass
            elif(arg == "arg_name"):
                name += "{} (".format(this_time)
            else: 
                if first: first = False
                else: name += ", "
                name += "{}: {}".format(arg, this_time)
    if(name == ""): name = "default" 
    else:           name += ")"
    return(name)

args.arg_title = get_args_title(default_args, args)

folder = "saved/" + args.arg_name
if(args.arg_name[:3] != "___" and not args.arg_name in ["default", "plotting"]):
    try: os.mkdir(folder)
    except: pass
if(default_args.alpha == "None"): default_args.alpha = None
if(args.alpha == "None"):         args.alpha = None

if(args == default_args): print("Using default arguments.", flush = True)
else:
    for arg in vars(default_args):
        default, this_time = getattr(default_args, arg), getattr(args, arg)
        if(this_time == default): pass
        else: print("{}:\n\tDefault:\t{}\n\tThis time:\t{}".format(arg, default, this_time), flush = True)



def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass
    
class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)


    
def dkl(mu_1, std_1, mu_2, std_2):
    std_1 = std_1**2
    std_2 = std_2**2
    term_1 = (mu_2 - mu_1)**2 / std_2 
    term_2 = std_1 / std_2 
    term_3 = torch.log(term_2)
    out = (.5 * (term_1 + term_2 - term_3 - 1))
    out = torch.nan_to_num(out)
    return(out)



from torch import nn 
def invert_linear_layer(layer):
    weights = layer.weight.data
    reverse_weights = torch.pinverse(weights)
    reverse_layer = nn.Linear(layer.out_features, layer.in_features)
    reverse_layer.weight.data = reverse_weights
    return(reverse_layer)
# %%
