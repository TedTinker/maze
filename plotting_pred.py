import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import imageio
from io import BytesIO
import os
import numpy as np

from utils import args, duration, load_dicts

print("name:\n{}".format(args.arg_name), flush = True)



def easy_plotting_pred(complete_order, plot_dicts):
    print("Easy not implemented yet")
            
            
        
def hard_plotting_pred(complete_order, plot_dicts):
    print("Hard not implemented yet")
    
    

plot_dicts, min_max_dict, (easy, complete_easy_order, easy_plot_dicts), (hard, complete_hard_order, hard_plot_dicts) = load_dicts(args)    
if(easy): print("\nPlotting predictions in easy maze.\n")    ; easy_plotting_pred(complete_easy_order, easy_plot_dicts)
if(hard): print("\nPlotting predictions in hard maze(s).\n") ; hard_plotting_pred(complete_hard_order, hard_plot_dicts)    
print("\nDuration: {}. Done!".format(duration()), flush = True)