#%% 
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest as ztest
from scipy.stats import binom_test
import scipy.stats as stats
from copy import deepcopy
import math

from utils import args, duration, load_dicts, print, maze_real_names

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

def format_num(num):
    if num == 0: return "<1e-323"
    elif abs(num) < 0.0001: return '{:.2e}'.format(num)
    elif abs(num) >= 0.0001:
        decimals = 3 - int(math.floor(math.log10(abs(num)))) # calculate number of decimal places needed
        return '{:.{}f}'.format(num, decimals)
    else: return str(num)
    
def fraction_to_float(fraction_str):
    numerator, denominator = map(int, fraction_str.split('/'))
    return numerator / denominator

def confidence_interval(fraction_str):
    numerator, denominator = map(int, fraction_str.split('/'))
    proportion = numerator / denominator
    standard_error = math.sqrt((proportion * (1 - proportion)) / denominator)
    z_score = stats.norm.ppf(1 - (1 - .95) / 2)
    margin_of_error = z_score * standard_error
    lower_bound = proportion - margin_of_error
    upper_bound = proportion + margin_of_error
    return(lower_bound, upper_bound)

def custom_round(number):
    str_num = str(number)
    integer_part, decimal_part = str_num.split('.')
    non_zero_pos = 0
    for char in decimal_part:
        if char != '0':
            break
        non_zero_pos += 1
    rounded_number = round(number, non_zero_pos + 1)
    return rounded_number
     


def hard_p_values(complete_order, plot_dicts):
    too_many_plot_dicts = len(plot_dicts) > 20
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
            
            # This line gives warning
            _, p = ztest([good_spots_1, good_spots_2], [len(spots_1), len(spots_2)])
            
            confidence = .999 
            good_p = "<{}".format(custom_round(1-confidence))
            if(p <= 1-confidence): 
                if(prop_1 < prop_2): color = "red"
                else:                color = "green"
            else:        color = "white"
            print("({}, {}),\t{} vs {}: \t{} vs {}, \tp={},\t{}.".format(x, y, arg_1, arg_2, prop_1, prop_2, p, color))
            
            p_value_dicts[(maze_name, epochs)][(arg_1, arg_2)] = [x, y, spots_1, spots_2, good_spots_1, good_spots_2, p, color]
            
        total_epochs += epochs
    
    total_epochs = 0
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
        plt.title("P-Values\n(Epoch {}, {})".format(epochs + total_epochs, maze_real_names[maze_name]))        
        plt.yticks(range(len(arg_names)), reversed_names, rotation='horizontal')
        plt.xticks(range(len(arg_names)), real_arg_names, rotation='vertical')
        plt.savefig("bad_p_value_plot.png".format(maze_name), format = "png", bbox_inches = "tight")
        plt.close()
        
        total_epochs += epochs
                
                
             
    kinds = ["d", "e", "n", "f", "en", "ef"]

    total_epochs = 0
    for (maze_name, epochs), p_value_dict in p_value_dicts.items():
        all_vals = []
        for kind in kinds:
            for (arg_1, arg_2), (x, y, spots_1, spots_2, good_spots_1, good_spots_2, p, color) in p_value_dict.items():
                base_arg_1 = arg_1.split("_")[0]
                if(arg_1.split("_")[-1] != "rand" and base_arg_1 == kind):
                    if(arg_1 in real_names): real_name_1 = real_names[arg_1]
                    all_vals.append([real_name_1, "{}/{}".format(good_spots_1, len(spots_1))])
                    break
        all_names = [val[0] for val in all_vals]
        all_heights = [fraction_to_float(val[1]) for val in all_vals]
        all_conf = [confidence_interval(val[1]) for val in all_vals]
        all_colors = ["white" for val in all_vals]
        fig, ax = plt.subplots()

        x = .1
        bar_width = 0.4  
        spacing = 0.5  

        for i in range(len(all_heights)):
            bar_center = x + bar_width / 2
            ax.add_patch(patches.Rectangle((x, 0), bar_width, all_heights[i], facecolor=all_colors[i], edgecolor="black"))
            ax.text(x + bar_width/2, -0.2, all_names[i], ha='center', va='center', rotation=90, fontsize=10)
            ax.plot([bar_center, bar_center], [all_conf[i][0], all_conf[i][1]], color="black")
            ax.plot([bar_center - 0.02, bar_center + 0.02], [all_conf[i][0], all_conf[i][0]], color="black")
            ax.plot([bar_center - 0.02, bar_center + 0.02], [all_conf[i][1], all_conf[i][1]], color="black")
            x += spacing  

        ax.set_xlim(0, x)
        ax.set_ylim(0, 1)  # Adding 1 for a bit of padding at the top
        ax.set_ylabel('Proportion of Good Exits')
        plt.title("Hypothesis 1\n(Epoch {}, {})".format(epochs + total_epochs, maze_real_names[maze_name])) 
        ax.axes.get_xaxis().set_visible(False)  # Hide the x-axis

        plt.savefig("{}_p_values_hypothesis_1.png".format(maze_name), format = "png", bbox_inches = "tight", dpi=300)
        plt.close()
        total_epochs += epochs
                
        
        
    total_epochs = 0
    for (maze_name, epochs), p_value_dict in p_value_dicts.items():
        all_vals = []
        for (arg_1, arg_2), (x, y, spots_1, spots_2, good_spots_1, good_spots_2, p, color) in p_value_dict.items():
            if(("n" in arg_1.split("_")[0] or "f" in arg_1.split("_")[0]) and arg_2 == arg_1 + "_rand"):
                if arg_1.endswith("rand"): real_name_1 = "with Curiosity Traps"
                elif(arg_1 in real_names): real_name_1 = real_names[arg_1]
                else:                      real_name_1 = arg_1
                if arg_2.endswith("rand"): real_name_2 = "with Curiosity Traps"
                elif(arg_2 in real_names): real_name_2 = real_names[arg_2]
                else:                      real_name_2 = arg_2
                all_vals.append([real_name_1, "{}/{}".format(good_spots_1, len(spots_1)), "{}/{}".format(good_spots_2, len(spots_2)), good_p if p < 1-confidence else str(p)])
        all_names = sum([[val[0], "with Curiosity Traps"] for val in all_vals], [])
        all_heights = sum([[fraction_to_float(val[1]), fraction_to_float(val[2])] for val in all_vals], [])
        all_conf = sum([[confidence_interval(val[1]), confidence_interval(val[2])] for val in all_vals], [])
        all_colors = sum([["white", "white"] if val[-1] != good_p else ["green", "red"] if fraction_to_float(val[1]) > fraction_to_float(val[2]) else ["red", "green"] for val in all_vals], [])
        fig, ax = plt.subplots()

        x = .1
        bar_width = 0.4  
        spacing = 0.5  

        for i in range(len(all_heights)):
            bar_center = x + bar_width / 2
            ax.add_patch(patches.Rectangle((x, 0), bar_width, all_heights[i], facecolor=all_colors[i], edgecolor="black"))
            ax.text(x + bar_width/2, -0.2, all_names[i], ha='center', va='center', rotation=90, fontsize=10)
            ax.plot([bar_center, bar_center], [all_conf[i][0], all_conf[i][1]], color="black")
            ax.plot([bar_center - 0.02, bar_center + 0.02], [all_conf[i][0], all_conf[i][0]], color="black")
            ax.plot([bar_center - 0.02, bar_center + 0.02], [all_conf[i][1], all_conf[i][1]], color="black")
            if(i % 2 != 0):
                x += spacing  
            else:
                x += bar_width

        ax.set_xlim(0, x)
        ax.set_ylim(0, 1)  # Adding 1 for a bit of padding at the top
        ax.set_ylabel('Proportion of Good Exits')
        plt.title("Hypothesis 2\n(Epoch {}, {})".format(epochs + total_epochs, maze_real_names[maze_name]))   
        ax.axes.get_xaxis().set_visible(False)  # Hide the x-axis

        plt.savefig("{}_p_values_hypothesis_2.png".format(maze_name), format = "png", bbox_inches = "tight", dpi=300)
        plt.close()
        total_epochs += epochs
        


plot_dicts, min_max_dict, (easy, complete_easy_order, easy_plot_dicts), (hard, complete_hard_order, hard_plot_dicts) = load_dicts(args)               
if(hard): print("\nPlotting p-values in hard maze(s).\n") ; hard_p_values(complete_hard_order, hard_plot_dicts)   
print("\nDuration: {}. Done!".format(duration()))
# %%
