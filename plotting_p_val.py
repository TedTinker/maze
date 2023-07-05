from itertools import product
import matplotlib.pyplot as plt

from utils import args, duration, load_dicts, print

print("name:\n{}\n".format(args.arg_name),)



def hard_p_values(complete_order, plot_dicts):
    rows = len(plot_dicts)
    arg_names = [arg_name for arg_name in complete_order if not arg_name in ["break", "empty_space"]]
    total_epochs = 0
    
    for maze_name, epochs in zip(plot_dicts[0]["args"].maze_list, plot_dicts[0]["args"].epochs):
        done_combos = []
        fig, axs = plt.subplots(rows, rows, figsize = (rows * 10, rows * 10))
        fig.suptitle("Epoch {} (Maze {})".format(epochs, maze_name), y = 1.05)
                
        for arg_1, arg_2 in product(arg_names, repeat = 2):
            if((arg_1,arg_2) in arg_names or (arg_2, arg_1) in arg_names): pass 
            else:
                arg_names.append((arg_1,arg_2))
                ax = axs[arg_names.index(arg_1), arg_names.index(arg_2)]
                for plot_dict in plot_dicts:
                    if(plot_dict["args"].name == arg_1): spots_1 = plot_dict["args"].spot_names
                    if(plot_dict["args"].name == arg_2): spots_2 = plot_dict["args"].spot_names 
                
                plt.savefig("{}.png".format(maze_name), format = "png", bbox_inches = "tight")
                plt.close()
                


plot_dicts, min_max_dict, (easy, complete_easy_order, easy_plot_dicts), (hard, complete_hard_order, hard_plot_dicts) = load_dicts(args)               
if(hard): print("\nPlotting P-values in hard maze(s).\n") ; hard_p_values(complete_hard_order, hard_plot_dicts)   
print("\nDuration: {}. Done!".format(duration()))