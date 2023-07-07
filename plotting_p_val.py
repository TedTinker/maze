from itertools import product
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import chi2_contingency
from copy import deepcopy
from scipy.stats import binom_test

from utils import args, duration, load_dicts, print

print("name:\n{}\n".format(args.arg_name),)



real_names = {
    "d"  : "No Entropy,\nNo Curiosity",
    "e"  : "Entropy",
    "n"  : "Naive Curiosity",
    "en" : "Entropy and\nNaive Curiosity",
    "f"  : "Aware Curiosity",
    "ef" : "Entropy and \nAware Curiosity",
}

def add_this(name):
    keys, values = [], []
    for key, value in real_names.items(): keys.append(key) ; values.append(value)
    for key, value in zip(keys, values):  
        new_key = key + "_" + name 
        real_names[new_key] = value
add_this("hard")
add_this("many")
add_this("1")
add_this("2")
add_this("3")
add_this("4")
add_this("5")
add_this("6")

real_maze_names = {
    "t" : "Biased T-Maze",
    "1" : "T-Maze",
    "2" : "Double T-Maze",
    "3" : "Triple T-Maze",
}
     


def hard_p_values(complete_order, plot_dicts):
    arg_names = [arg_name for arg_name in complete_order if not arg_name in ["break", "empty_space"]]
    
    def custom_sort(item):
        if item.endswith("_rand"): return(arg_names.index(item[:-5]))
        else:                      return(arg_names.index(item))

    arg_names = sorted(arg_names, key=custom_sort)
    real_arg_names = [real_names[arg_name] if not arg_name.endswith("rand") else "with Curiosity Trap" for arg_name in arg_names]
    reversed_names = deepcopy(real_arg_names)
    reversed_names.reverse()
    total_epochs = 0
    
    for maze_name, epochs in zip(plot_dicts[0]["args"].maze_list, plot_dicts[0]["args"].epochs):
        done_combos = []
        plt.figure(figsize = (10, 10))
        plt.xlim([-.5, len(arg_names)-.5])
        plt.ylim([-.5, len(arg_names)-.5])
        plt.title("P-Values\n(Epoch {}, Maze {})".format(epochs + total_epochs, real_maze_names[maze_name]))        
        plt.yticks(range(len(arg_names)), reversed_names, rotation='horizontal')
        plt.xticks(range(len(arg_names)), real_arg_names, rotation='vertical')

        for (x, arg_1), (y, arg_2) in product(enumerate(arg_names), repeat = 2):
            if(x == y):
                p = "" ; color = "black"
            else:
                done_combos.append((arg_1,arg_2))
                for plot_dict in plot_dicts:
                    if(plot_dict["args"].arg_name == arg_1): spots_1 = [spot_names[epochs + total_epochs - 1] for spot_names in plot_dict["spot_names"]]
                    if(plot_dict["args"].arg_name == arg_2): spots_2 = [spot_names[epochs + total_epochs - 1] for spot_names in plot_dict["spot_names"]]
                
                spots_1 = ["good" if spot in ["RIGHT", "LEFT\nLEFT", "RIGHT\nLEFT\nLEFT"] else "bad" for spot in spots_1]
                spots_2 = ["good" if spot in ["RIGHT", "LEFT\nLEFT", "RIGHT\nLEFT\nLEFT"] else "bad" for spot in spots_2]
                
                good_spots_1 = spots_1.count("good")
                good_spots_2 = spots_2.count("good")

                total_spots_1 = len(spots_1)
                total_spots_2 = len(spots_2)

                prop_1 = good_spots_1 / total_spots_1
                prop_2 = good_spots_2 / total_spots_2

                p = binom_test(min(good_spots_1, good_spots_2), n=max(total_spots_1, total_spots_2), p=max(prop_1, prop_2), alternative='less')

                if(p < .05): 
                    if(prop_1 > prop_2): color = "red"
                    else:                color = "green"
                else:        color = "white"
                p = "{}".format(round(p,2))
            
            y = -1*y + len(arg_names) - 1
            plt.gca().add_patch(patches.Rectangle((x-.5, y-.5), 1, 1, facecolor=color))
            plt.text(x, y, p, fontsize=12, ha='center', va='center')
        
        plt.savefig("{}_p_values.png".format(maze_name), format = "png", bbox_inches = "tight")
        plt.close()
                


plot_dicts, min_max_dict, (easy, complete_easy_order, easy_plot_dicts), (hard, complete_hard_order, hard_plot_dicts) = load_dicts(args)               
if(hard): print("\nPlotting p-values in hard maze(s).\n") ; hard_p_values(complete_hard_order, hard_plot_dicts)   
print("\nDuration: {}. Done!".format(duration()))