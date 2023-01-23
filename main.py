#%%

import os
from shutil import rmtree
import pickle
import argparse

from utils import default_args
from plotting import plots
from train import Trainer


    
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



def save_title(args, title):
    trainer = Trainer(args, title)
    plot_dict, min_max_dict = trainer.train()
    folder = "saved/{}".format(title)
    if(os.path.isdir(folder)): rmtree(folder)
    os.mkdir(folder)
    with open(folder + "/args.pickle", "wb") as handle:
        pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder + "/plot_dict.pickle", "wb") as handle:
        pickle.dump(plot_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder + "/min_max_dict.pickle", "wb") as handle:
        pickle.dump(min_max_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
        
def load_title(title):
    folder = "saved/{}".format(title)
    with open(folder + "/args.pickle", "rb") as handle:
        args = pickle.load(handle)
    with open(folder + "/plot_dict.pickle", "rb") as handle:
        plot_dict = pickle.load(handle)
    with open(folder + "/min_max_dict.pickle", "rb") as handle:
        min_max_dict = pickle.load(handle)
    return(args, plot_dict, min_max_dict)


                
def get_plots(arg_dict_list):
    plot_dicts = [] ; min_max_dicts = [] 
    for i, arg_dict in enumerate(arg_dict_list):
        args, title = get_title(arg_dict)
        try: 
            _, plot_dict, min_max_dict = load_title(title)
            print("{} loaded.".format(title))
        except: 
            save_title(args, title)
            _, plot_dict, min_max_dict = load_title(title)
        plot_dict["title"] = "{}: ".format(i) + plot_dict["title"]
        plot_dicts.append(plot_dict) ; min_max_dicts.append(min_max_dict)
        
        for key in min_max_dict.keys():
            if(not key in ["title", "spot_names"]):
                minimum = None ; maximum = None 
                for mm_dict in min_max_dicts:
                    if(mm_dict[key] != (None, None)):
                        if(minimum == None):             minimum = mm_dict[key][0]
                        elif(minimum > mm_dict[key][0]): minimum = mm_dict[key][0]
                        if(maximum == None):             maximum = mm_dict[key][1]
                        elif(maximum < mm_dict[key][1]): maximum = mm_dict[key][1]
                min_max_dict[key] = (minimum, maximum)
        
        plots(plot_dicts, min_max_dict)



get_plots([
    {},
    {"alpha" : None},
    {                "eta" : 1},
    {"alpha" : None, "eta" : 1},
    {"alpha" : None, "eta" : .0001, "naive" : False},
    {"alpha" : None, "eta" : .01,   "naive" : False, "dkl_change_size" : "step"}
])

# %%
