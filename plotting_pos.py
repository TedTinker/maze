import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import imageio
from io import BytesIO

from utils import duration



def easy_plotting_pos(complete_order, plot_dicts):
    images = []
    rows = 0 ; columns = 0 ; current_count = 0 
    for arg_name in complete_order: 
        if(arg_name == "break"): rows += 1 ; columns = max(columns, current_count) ; current_count = 0
        else: current_count += 1
    columns = max(columns, current_count)
    if(complete_order[-1] != "break"): rows += 1
    
    epochs = list(set([int(key.split("_")[1]) for key in plot_dicts[0]["pos_lists"].keys()])) ; epochs.sort()
    steps = len(plot_dicts[0]["pos_lists"]["0_0"][0])
    agents = list(set([int(key.split("_")[0]) for key in plot_dicts[0]["pos_lists"].keys()])) ; agents.sort()
    episodes = len(plot_dicts[0]["pos_lists"]["0_0"])
    
    cmap = plt.cm.get_cmap("gray_r")
    norm = Normalize(vmin=0, vmax=1)
    handles = []
    for c in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:
        handle = plt.scatter(0, 0, marker = "s", s = 250, facecolor = cmap(norm(c)))
        handles.append(handle)
    plt.close()
    
    for e in epochs:
        for s in range(steps):
            fig, axs = plt.subplots(rows, columns, figsize = (columns * 3, rows * 1.5))
            fig.suptitle("Epoch {} Step {}".format(e, s), y = 1.1)
            plot_position = (0, 0)
            for arg_name in complete_order:
                if(  arg_name == "break"):       plot_position = (plot_position[0] + 1, 0)
                elif(arg_name == "empty_space"): 
                    ax = axs[plot_position[0], plot_position[1]] if rows > 1 else axs[plot_position[1]]
                    ax.axis("off")
                    plot_position = (plot_position[0],     plot_position[1] + 1)
                else:
                    for plot_dict in plot_dicts:
                        if(plot_dict["arg_name"] == arg_name): break

                    ax = axs[plot_position[0], plot_position[1]] if rows > 1 else axs[plot_position[1]]
                    for spot in [(0, 0), (0, 1), (-1, 1), (1, 1), (1, 2), (2, 2), (3, 2), (3, 1)]:
                        ax.text(spot[0], spot[1], "\u25A1", fontsize = 30, ha = "center", va = "center")
                    to_plot = {}
                    for a in agents:
                        for ep in range(episodes):
                            coordinate = plot_dict["pos_lists"]["{}_{}".format(a, e)][ep][s]
                            if(not coordinate in to_plot): to_plot[coordinate] = 0
                            to_plot[coordinate] += 1
                            
                    total_points = sum(to_plot.values())
                    to_plot = {k: v / total_points for k, v in to_plot.items()}
                    coords, proportions = zip(*to_plot.items())
                    x_coords, y_coords = zip(*coords)
                    
                    if(len(proportions) == 1): ax.scatter(x_coords, y_coords, marker = "s", s = 250, cmap = cmap, c = "black")
                    else:                      ax.scatter(x_coords, y_coords, marker = "s", s = 250, cmap = cmap, c = proportions, norm = norm)
                    ax.set_ylim([-.5, 2.5])
                    ax.set_xlim([-1.5, 3.5])
                    ax.set_title("{}".format(plot_dict["arg_name"]))
                    ax.axis("off")
                
                    plot_position = (plot_position[0], plot_position[1] + 1)
                
            fig.legend(loc = "upper left", handles = handles, labels= ["{}%".format(p) for p in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]])
            buf = BytesIO()
            plt.savefig(buf, format = "png", bbox_inches = "tight")
            buf.seek(0)
            im = imageio.imread(buf)
            images.append(im)
            buf.close()
            plt.close()
            
        print("Done with epoch {}:\t{}.".format(e, duration()), flush = True)
            
    imageio.mimwrite("video.mp4", images, fps = 3)
            
            
        
def hard_plotting_pos(complete_order, plot_dicts):
    pass