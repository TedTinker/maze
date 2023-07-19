#%% 
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest as ztest
from scipy.stats import binom_test
from copy import deepcopy
import math

from utils import args, duration, load_dicts, print

print("name:\n{}\n".format(args.arg_name),)



real_names = {
    "d"  : "No Entropy,\nNo Curiosity",
    "e"  : "Entropy",
    "n"  : "Naive Curiosity",
    "en" : "Entropy and\nNaive Curiosity",
    "f"  : "Aware Curiosity",
    "ef" : "Entropy and \nAware Curiosity"}

def add_this(dict, name):
    keys, values = [], []
    for key, value in dict.items(): keys.append(key) ; values.append(value)
    for key, value in zip(keys, values):  
        new_key = key + "_" + name 
        dict[new_key] = value
add_this(real_names, "hard")
add_this(real_names, "many")
for i in range(15):
    add_this(real_names, str(i))

real_maze_names = {
    "t" : "Biased T-Maze",
    "1" : "T-Maze",
    "2" : "Double T-Maze",
    "3" : "Triple T-Maze"}

def format_num(num):
    if num == 0: return "<1e-323"
    elif abs(num) < 0.0001: return '{:.2e}'.format(num)
    elif abs(num) >= 0.0001:
        decimals = 3 - int(math.floor(math.log10(abs(num)))) # calculate number of decimal places needed
        return '{:.{}f}'.format(num, decimals)
    else: return str(num)
     


def hard_p_values(complete_order, plot_dicts):
    arg_names = [arg_name for arg_name in complete_order if not arg_name in ["break", "empty_space"]]
    
    def custom_sort(item):
        if item.endswith("_rand"): return(arg_names.index(item[:-5]))
        else:                      return(arg_names.index(item))

    arg_names = sorted(arg_names, key=custom_sort)
    real_arg_names = []
    for arg_name in arg_names:
        if arg_name.endswith("rand"): real_name = "with Curiosity Traps"
        elif(arg_name in real_names): real_name = real_names[arg_name]
        else:                         real_name = arg_name
        real_arg_names.append(real_name)
    reversed_names = deepcopy(real_arg_names)
    reversed_names.reverse()
    total_epochs = 0
    
    p_value_dicts = {}
    for maze_name, epochs in zip(plot_dicts[0]["args"].maze_list, plot_dicts[0]["args"].epochs):
        p_value_dicts[(maze_name, epochs)] = {}

        # Maybe use cumulative rewards
        for (x, arg_1), (y, arg_2) in product(enumerate(arg_names), repeat = 2):
            for plot_dict in plot_dicts:
                if(plot_dict["args"].arg_name == arg_1): spots_1 = sum([spot_names[epochs + total_epochs - 11 : epochs + total_epochs - 1] for spot_names in plot_dict["spot_names"]], [])
                if(plot_dict["args"].arg_name == arg_2): spots_2 = sum([spot_names[epochs + total_epochs - 11 : epochs + total_epochs - 1] for spot_names in plot_dict["spot_names"]], [])
            
            spots_1 = ["good" if spot in ["RIGHT", "LEFT\nLEFT", "RIGHT\nLEFT\nLEFT"] else "bad" for spot in spots_1]
            spots_2 = ["good" if spot in ["RIGHT", "LEFT\nLEFT", "RIGHT\nLEFT\nLEFT"] else "bad" for spot in spots_2]
            
            good_spots_1 = spots_1.count("good")
            good_spots_2 = spots_2.count("good")

            total_spots_1 = len(spots_1)
            total_spots_2 = len(spots_2)

            prop_1 = good_spots_1 / total_spots_1
            prop_2 = good_spots_2 / total_spots_2

            _, p = ztest([good_spots_1, good_spots_2], [len(spots_1), len(spots_2)])
            
            confidence = .999 
            if(p <= 1-confidence): 
                if(prop_1 < prop_2): color = "red"
                else:                color = "green"
            else:        color = "white"
            print("({}, {}),\t{} vs {}: \t{} vs {}, \tp={},\t{}.".format(x, y, arg_1, arg_2, prop_1, prop_2, p, color))
            
            p_value_dicts[(maze_name, epochs)][(arg_1, arg_2)] = [x, y, spots_1, spots_2, good_spots_1, good_spots_2, p, color]
    
    
    
    for (maze_name, epochs), p_value_dict in p_value_dicts.items():
        plt.figure(figsize = (10, 10))
        ax = plt.gca()
        for (arg_1, arg_2), (x, y, spots_1, spots_2, good_spots_1, good_spots_2, p, color) in p_value_dict.items():
            flipped_y = -1*y + len(arg_names) - 1
            if(x > y): pass
            else:
                if(x == y): 
                    plt.gca().add_line(Line2D([x - .5, x + .5], [flipped_y + .5, flipped_y - .5], color='black', linewidth = .5))
                else:  
                    plt.gca().add_patch(patches.Rectangle((x - .5, flipped_y - .5), 1, 1, facecolor=color))
                    plt.text(x, flipped_y, round(p,2), fontsize=12, ha='center', va='center')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlim([-.5, len(arg_names)-.5])
        plt.ylim([-.5, len(arg_names)-.5])
        plt.title("P-Values\n(Epoch {}, {})".format(epochs + total_epochs, real_maze_names[maze_name]))        
        plt.yticks(range(len(arg_names)), reversed_names, rotation='horizontal')
        plt.xticks(range(len(arg_names)), real_arg_names, rotation='vertical')
        plt.savefig("bad_p_value_plot.png".format(maze_name), format = "png", bbox_inches = "tight")
        plt.close()
                
                
             
    pairs = {
        "d" : ["e", "n", "f"],
        "e" : ["en", "ef"],
        "n" : ["en"],
        "f" : ["ef"]}

    for (maze_name, epochs), p_value_dict in p_value_dicts.items():
        data = [["", "Good Exits", "", "Good Exits", "P-Value"]]     
        p_vals = []
        for (arg_1, arg_2), (x, y, spots_1, spots_2, good_spots_1, good_spots_2, p, color) in p_value_dict.items():
            base_arg_1, base_arg_2 = arg_1.split("_")[0], arg_2.split("_")[0]
            if(arg_1.split("_")[-1] != "rand" and arg_2.split("_")[-1] != "rand"):
                if(base_arg_1 in list(pairs.keys())):
                    if(base_arg_2 in pairs[base_arg_1]):
                        if arg_1.endswith("rand"): real_name_1 = "with Curiosity Traps"
                        elif(arg_1 in real_names): real_name_1 = real_names[arg_1]
                        else:                      real_name_1 = arg_1
                        if arg_2.endswith("rand"): real_name_2 = "with Curiosity Traps"
                        elif(arg_2 in real_names): real_name_2 = real_names[arg_2]
                        else:                      real_name_2 = arg_2
                        p_vals.append(p)
                        data.append([real_name_1, "{}/{}".format(good_spots_1, len(spots_1)), real_name_2, "{}/{}".format(good_spots_2, len(spots_2)), format_num(p)])
        plt.figure(figsize = (10, 7))
        ax = plt.gca() ; ax.axis("off")
        plt.title("P-Values: Hypothesis 1\n(Epoch {}, {})".format(epochs + total_epochs, real_maze_names[maze_name]))   
        table = plt.table(cellText=data, loc='center', cellLoc='center', colWidths=[.3, .2, .3, .2, .2])
        cells = table.get_celld()
        for i in range(1, len(data)):
            p_value = p_vals[i-1]
            col_1 = int(data[i][1].split("/")[0])
            col_2 = int(data[i][3].split("/")[0])
            if p_value <= 0.05:  
                if(col_1 > col_2): cells[i, 1].set_facecolor('green') ; cells[i, 3].set_facecolor('red')
                else:              cells[i, 1].set_facecolor('red') ;   cells[i, 3].set_facecolor('green')

        table.scale(1, 4)
        plt.savefig("{}_p_values_hypothesis_1.png".format(maze_name), format = "png", bbox_inches = "tight")
        plt.close()
             
             
                
    for (maze_name, epochs), p_value_dict in p_value_dicts.items():
        data = [["", "Good Exits", "with Traps", "P-Value"]]     
        p_vals = []
        for (arg_1, arg_2), (x, y, spots_1, spots_2, good_spots_1, good_spots_2, p, color) in p_value_dict.items():
            if(("n" in arg_1.split("_")[0] or "f" in arg_1.split("_")[0]) and arg_2 == arg_1 + "_rand"):
                if arg_1.endswith("rand"): real_name_1 = "with Curiosity Traps"
                elif(arg_1 in real_names): real_name_1 = real_names[arg_1]
                else:                      real_name_1 = arg_1
                if arg_2.endswith("rand"): real_name_2 = "with Curiosity Traps"
                elif(arg_2 in real_names): real_name_2 = real_names[arg_2]
                else:                      real_name_2 = arg_2
                p_vals.append(p)
                data.append([real_name_1, "{}/{}".format(good_spots_1, len(spots_1)), "{}/{}".format(good_spots_2, len(spots_2)), format_num(p)])
        plt.figure(figsize = (10, 5))
        ax = plt.gca() ; ax.axis("off")
        plt.title("P-Values: Hypothesis 2\n(Epoch {}, {})".format(epochs + total_epochs, real_maze_names[maze_name]))   
        table = plt.table(cellText=data, loc='center', cellLoc='center', colWidths=[.3, .2, .2, .2])
        cells = table.get_celld()
        for i in range(1, len(data)):
            p_value = p_vals[i-1] 
            col_1 = int(data[i][1].split("/")[0])
            col_2 = int(data[i][2].split("/")[0])
            if p_value <= 0.05:  
                if(col_1 > col_2): cells[i, 1].set_facecolor('green') ; cells[i, 2].set_facecolor('red')
                else:              cells[i, 1].set_facecolor('red') ;   cells[i, 2].set_facecolor('green')

        table.scale(1, 4)
        plt.savefig("{}_p_values_hypothesis_2.png".format(maze_name), format = "png", bbox_inches = "tight")
        plt.close()
        


plot_dicts, min_max_dict, (easy, complete_easy_order, easy_plot_dicts), (hard, complete_hard_order, hard_plot_dicts) = load_dicts(args)               
if(hard): print("\nPlotting p-values in hard maze(s).\n") ; hard_p_values(complete_hard_order, hard_plot_dicts)   
print("\nDuration: {}. Done!".format(duration()))
# %%
