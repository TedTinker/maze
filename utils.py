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



import argparse, ast, os, pickle
from math import exp, pi
import numpy as np
from time import sleep

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
parser.add_argument('--randomness',         type=int,   default = 0)

# Hard Maze
parser.add_argument('--body_size',          type=float, default = 2)    
parser.add_argument('--image_size',         type=int,   default = 8)
parser.add_argument('--min_speed',          type=float, default = 50)
parser.add_argument('--max_speed',          type=float, default = 100)
parser.add_argument('--steps_per_step',     type=int,   default = 5)
parser.add_argument('--max_yaw_change',     type=float, default = pi/2)

# Module 
parser.add_argument('--hidden_size',        type=int,   default = 32)   
parser.add_argument('--state_size',         type=int,   default = 32)
parser.add_argument('--forward_lr',         type=float, default = .01)
parser.add_argument('--alpha_lr',           type=float, default = .01) 
parser.add_argument('--actor_lr',           type=float, default = .01)
parser.add_argument('--critic_lr',          type=float, default = .01)
parser.add_argument('--action_prior',       type=str,   default = "normal")
parser.add_argument("--tau",                type=float, default = .05)      # For soft-updating target critics

# Complexity 
parser.add_argument('--std_min',            type=int,   default = exp(-20))
parser.add_argument('--std_max',            type=int,   default = exp(2))
parser.add_argument("--beta_obs",           type=float, default = 2)
parser.add_argument("--beta_zq",            type=float, default = 2)      
parser.add_argument("--sigma_obs",          type=float, default = 1)     
parser.add_argument("--sigma_zq",           type=float, default = 1)      

# Memory buffer
parser.add_argument('--capacity',           type=int,   default = 100)

# Training
parser.add_argument('--epochs',             type=tuple_type,   default = (3000,))
parser.add_argument('--steps_per_epoch',    type=int,   default = 10)
parser.add_argument('--batch_size',         type=int,   default = 8)
parser.add_argument('--GAMMA',              type=int,   default = .99)
parser.add_argument("--d",                  type=int,   default = 2)        # Delay to train actors
parser.add_argument('--accuracy',           type=str,   default = "mse")

# Entropy
parser.add_argument("--alpha",              type=str,   default = 0)        # Soft-Actor-Critic entropy aim
parser.add_argument("--target_entropy",     type=float, default = -2)       # Soft-Actor-Critic entropy aim

# Curiosity
parser.add_argument("--curiosity",          type=str,   default = "none")     # Which kind of curiosity
parser.add_argument("--naive_eta_obs",      type=float, default = .15)        # Scale curiosity
parser.add_argument("--naive_eta_state",    type=float, default = .15)        # Scale curiosity
parser.add_argument("--free_eta_obs",       type=float, default = 3)        # Scale curiosity
parser.add_argument("--free_eta_state",     type=float, default = 3)        # Scale curiosity
parser.add_argument("--dkl_max",            type=float, default = 1)        

# Saving data
parser.add_argument('--keep_data',           type=int,   default = 25)
parser.add_argument('--epochs_per_pred_list',type=int,   default = 500)
parser.add_argument('--agents_per_pred_list',type=int,   default = 2)
parser.add_argument('--episodes_in_pred_list',type=int,  default = 2)
parser.add_argument('--samples_per_pred',    type=int,   default = 3)

parser.add_argument('--epochs_per_pos_list', type=int,   default = 500)
parser.add_argument('--agents_per_pos_list', type=int,   default = -1)
parser.add_argument('--episodes_in_pos_list',type=int,   default = 3)



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



args_not_in_title = ["arg_title", "id", "agents", "previous_agents", "hard_maze", "maze_list", "keep_data", "epochs_per_pred_list", "episodes_in_pred_list", "agents_per_pred_list", "epochs_per_pos_list", "episodes_in_pos_list", "agents_per_pos_list"]
def get_args_title(default_args, args):
    if(args.arg_title[:3] == "___"): return(args.arg_title)
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
    if(name.endswith(" ()")): name = name[:-3]
    return(name)

args.arg_title = get_args_title(default_args, args)

folder = "saved/" + args.arg_name
if(args.arg_title[:3] != "___" and not args.arg_name in ["default", "finishing_dictionaries", "plotting", "plotting_predictions", "plotting_positions"]):
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



def load_dicts(args):
    if(os.getcwd().split("/")[-1] != "saved"): os.chdir("saved")
    plot_dicts = [] ; min_max_dicts = []
        
    complete_order = args.arg_title[3:-3].split("+")
    order = [o for o in complete_order if not o in ["empty_space", "break"]]

    for name in order:
        got_plot_dicts = False ; got_min_max_dicts = False
        while(not got_plot_dicts):
            try:
                with open(name + "/" + "plot_dict.pickle", "rb") as handle: 
                    plot_dicts.append(pickle.load(handle)) ; got_plot_dicts = True
            except: print("Stuck trying to get {}'s plot_dicts...".format(name), flush = True) ; sleep(1)
        while(not got_min_max_dicts):
            try:
                with open(name + "/" + "min_max_dict.pickle", "rb") as handle: 
                    min_max_dicts.append(pickle.load(handle)) ; got_min_max_dicts = True 
            except: print("Stuck trying to get {}'s min_max_dicts...".format(name), flush = True) ; sleep(1)
            
    min_max_dict = {}
    for key in plot_dicts[0].keys():
        if(not key in ["args", "arg_title", "arg_name", "pred_lists", "pos_lists", "spot_names"]):
            minimum = None ; maximum = None
            for mm_dict in min_max_dicts:
                if(mm_dict[key] != (None, None)):
                    if(minimum == None):             minimum = mm_dict[key][0]
                    elif(minimum > mm_dict[key][0]): minimum = mm_dict[key][0]
                    if(maximum == None):             maximum = mm_dict[key][1]
                    elif(maximum < mm_dict[key][1]): maximum = mm_dict[key][1]
            min_max_dict[key] = (minimum, maximum)
            
    complete_easy_order = [] ; easy_plot_dicts = []
    complete_hard_order = [] ; hard_plot_dicts = []

    easy = False 
    hard = False 
    for arg_name in complete_order: 
        if(arg_name in ["break", "empty_space"]): 
            complete_easy_order.append(arg_name)
            complete_hard_order.append(arg_name)
        else:
            for plot_dict in plot_dicts:
                if(plot_dict["args"].arg_name == arg_name):    
                    if(plot_dict["args"].hard_maze): complete_hard_order.append(arg_name) ; hard_plot_dicts.append(plot_dict) ; hard = True
                    else:                            complete_easy_order.append(arg_name) ; easy_plot_dicts.append(plot_dict) ; easy = True
                    
    while len(complete_easy_order) > 0 and complete_easy_order[0] == "break": complete_easy_order.pop(0)
    while len(complete_hard_order) > 0 and complete_hard_order[0] == "break": complete_hard_order.pop(0)              
            
    return(plot_dicts, min_max_dict, (easy, complete_easy_order, easy_plot_dicts), (hard, complete_hard_order, hard_plot_dicts))
# %%
