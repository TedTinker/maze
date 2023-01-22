#%%

from copy import deepcopy
import os
from shutil import rmtree
import pickle

from utils import default_args
from plotting import plots
from train import Trainer

# To do: Have it save everything in a folder whose name is based on the args.
# That'll save time.



def get_title(bayes = True, dkl_rate = .001, entropy = -2, curiosity = 1, naive = True):
    args = deepcopy(default_args)
    args.bayes = bayes
    args.dkl_rate = dkl_rate
    if(entropy == False): args.alpha = 0
    else:                 args.alpha = None ; args.target_entropy = entropy
    if(curiosity == False):  args.eta = 0
    else:                    args.eta = curiosity ; args.naive_curiosity = naive
    title = ""
    first = True
    for arg in vars(args):
        if(getattr(args, arg) != getattr(default_args, arg)):
            if(not first): title += "_"
            title += "{}_{}".format(arg, getattr(args, arg)) ; first = False
    if(len(title) == 0): title = "default"
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


                
def get_plots(bayes_list, dkl_rate_list, entropy_list, curiosity_list, naive_list):
    plot_dicts = [] ; min_max_dicts = [] 
    num = min([len(bayes_list), len(dkl_rate_list), len(entropy_list), len(curiosity_list), len(naive_list)])
    for i in range(num):
        args, title = get_title(bayes_list[i], dkl_rate_list[i], entropy_list[i], curiosity_list[i], naive_list[i])
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



bayes_list     = [True] * 6 
dkl_rate_list  = [.001] * 6
entropy_list   = [False, -2,    False, -2,   -2]
curiosity_list = [False, False, 1,     1,    .0001]
naive_list     = [True,  True,  True,  True, False]

get_plots(bayes_list, dkl_rate_list, entropy_list, curiosity_list, naive_list)

# %%
