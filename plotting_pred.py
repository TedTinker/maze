import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import imageio
from io import BytesIO
import os
import numpy as np

from torch.distributions import Normal

from utils import args, duration, load_dicts

print("name:\n{}".format(args.arg_name), flush = True)



def easy_plotting_pred(complete_order, plot_dicts):
    epochs = list(set([int(key.split("_")[1]) for key in plot_dicts[0]["pred_lists"].keys()])) ; epochs.sort()
    agents = list(set([int(key.split("_")[0]) for key in plot_dicts[0]["pred_lists"].keys()])) ; agents.sort()
    episodes = len(plot_dicts[0]["pred_lists"]["0_0"])
    
    cmap = plt.cm.get_cmap("gray_r")
    norm = Normalize(vmin = -1, vmax = 1)
    handles = []
    for c in [-1, -.8, -.6, -.4, -.2, 0, .2, .4, .6, .8, 1]:
        handle = plt.scatter(0, 0, marker = "s", s = 250, facecolor = cmap(norm(c)))
        handles.append(handle)
    plt.close()
        
    for epoch in epochs:
        for agent in agents:
            for arg_name in complete_order:
                if(arg_name in ["break", "empty_space"]): pass 
                else:
                    for plot_dict in plot_dicts:
                        if(plot_dict["arg_name"] == arg_name): pred_lists = plot_dict["pred_lists"]["{}_{}".format(agent, epoch)] ; break 
                    obs_size = 12 + plot_dict["args"].randomness
                    for episode in range(episodes):
                        pred_list = pred_lists[episode]
                        rows = len(pred_list) ; columns = 1 + plot_dict["args"].samples_per_pred
                        fig, axs = plt.subplots(rows, columns, figsize = (columns * 3, rows * 1.5))
                        title = "Agent {}: Epoch {}, Episode {}".format(agent, epoch, episode)
                        fig.suptitle(title, y = 1.1)
                        for row, (obs, mu, std) in enumerate(pred_list):
                            for column in range(columns):
                                ax = axs[row, column] ; ax.axis("off")
                                if(row == 0 and column > 0): pass
                                else:                
                                    if(column == 0): 
                                        ax.scatter([x for x in range(obs_size)], [0 for _ in range(obs_size)], marker = "s", s = 250, linewidths = 1, edgecolor='black', cmap = cmap, c = obs, norm = norm)
                                        ax.set_title("Step {}".format(row))
                                    else:
                                        e = Normal(0, 1).sample(std.shape) ; pred = mu + e * std
                                        ax.scatter([x for x in range(obs_size)], [0 for _ in range(obs_size)], marker = "s", s = 250, linewidths = 1, edgecolor='black', cmap = cmap, c = pred, norm = norm)
                                        ax.set_title("{}".format("Step {}".format(row) if column == 0 else "Prediction Sample {}".format(column)))
                        plt.savefig("{}/{}.png".format(arg_name, title), format = "png", bbox_inches = "tight")
                        plt.close()
                                
                                                

            
        
def hard_plotting_pred(complete_order, plot_dicts):
    print("Hard not implemented yet")
    
    

plot_dicts, min_max_dict, (easy, complete_easy_order, easy_plot_dicts), (hard, complete_hard_order, hard_plot_dicts) = load_dicts(args)    
if(easy): print("\nPlotting predictions in easy maze.\n")    ; easy_plotting_pred(complete_easy_order, easy_plot_dicts)
if(hard): print("\nPlotting predictions in hard maze(s).\n") ; hard_plotting_pred(complete_hard_order, hard_plot_dicts)    
print("\nDuration: {}. Done!".format(duration()), flush = True)