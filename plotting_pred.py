import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import imageio
from io import BytesIO
import os, pickle
import numpy as np
from time import sleep

from utils import args, duration



def easy_plotting_pred(complete_order, plot_dicts):
    print("Easy not implemented yet")
            
            
        
def hard_plotting_pred(complete_order, plot_dicts):
    print("Hard not implemented yet")
    
    

os.chdir("saved")
plot_dicts = [] ; min_max_dicts = []
    
complete_order = args.arg_name[3:-3].split("+")
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
    if(not key in ["args", "arg_title", "arg_name", "pred_dicts", "pos_lists", "spot_names"]):
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

if(easy): print("\nPlotting predictions in easy maze.\n")    ; easy_plotting_pred(complete_easy_order, easy_plot_dicts)
if(hard): print("\nPlotting predictions in hard maze(s).\n") ; hard_plotting_pred(complete_hard_order, hard_plot_dicts)    

print("\nDuration: {}. Done!".format(duration()), flush = True)