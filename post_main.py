#%%

import os
import pickle
from time import sleep

from utils import args
from plotting import plots

print("name:\n{}".format(args.name))

plot_dicts = [] ; min_max_dicts = []
    
order = args.name[3:-3]
order = order.split("+")
order = [o for o in order if o != "break"]

for name in order:
    got_plot_dicts = False ; got_min_max_dicts = False
    while(not got_plot_dicts):
        try:
            with open("saved/" + name + "/" + "plot_dict.pickle", "rb") as handle: 
                plot_dicts.append(pickle.load(handle)) ; got_plot_dicts = True
        except: print("Stuck trying to get {}'s plot_dicts...".format(name)) ; sleep(1)
    while(not got_min_max_dicts):
        try:
            with open("saved/" + name + "/" + "min_max_dict.pickle", "rb") as handle: 
                min_max_dicts.append(pickle.load(handle)) ; got_min_max_dicts = True 
        except: print("Stuck trying to get {}'s min_max_dicts...".format(name)) ; sleep(1)
        
min_max_dict = {}
for key in plot_dicts[0].keys():
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
print("Done with {}!".format(args.name))