import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import imageio
from io import BytesIO
import os, pickle
import numpy as np
from time import sleep

from utils import args, duration



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
    norm = Normalize(vmin = 0, vmax = 1)
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
                if(  arg_name == "break"): plot_position = (plot_position[0] + 1, 0)
                elif(arg_name == "empty_space"): 
                    ax = axs[plot_position[0], plot_position[1]] if rows > 1 else axs[plot_position[1]]
                    ax.axis("off")
                    plot_position = (plot_position[0], plot_position[1] + 1)
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
                
            fig.legend(loc = "upper left", handles = handles, labels= ["{}%".format(p) for p in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]])
            buf = BytesIO()
            plt.savefig(buf, format = "png", bbox_inches = "tight")
            buf.seek(0)
            im = imageio.imread(buf)
            images.append(im)
            buf.close()
            plt.close()
            
        print("Done with epoch {}:\t{}.".format(e, duration()), flush = True)
            
    imageio.mimwrite("easy_video.mp4", images, fps = 3)
            
            
        
def hard_plotting_pos(complete_order, plot_dicts):
    os.chdir("..")
    images = []
    rows = 0 ; columns = 0 ; current_count = 0 
    for arg_name in complete_order: 
        if(arg_name == "break"): rows += 1 ; columns = max(columns, current_count) ; current_count = 0
        else: current_count += 1
    columns = max(columns, current_count)
    if(complete_order[-1] != "break"): rows += 1
    
    epochs = list(set([int(key.split("_")[1]) for key in plot_dicts[0]["pos_lists"].keys()])) ; epochs.sort()
    agents = list(set([int(key.split("_")[0]) for key in plot_dicts[0]["pos_lists"].keys()])) ; agents.sort()
    episodes = len(plot_dicts[0]["pos_lists"]["0_0"])
    maze_names = [plot_dicts[0]["pos_lists"]["{}_{}".format(0, e)][0][0] for e in epochs]
            
    cmap = plt.cm.get_cmap("hsv", len(agents))
    norm = Normalize(vmin = 0, vmax = len(agents))
    handles = []
    for c in agents:
        handle = plt.scatter(c, 0, marker = "s", s = 10, color=cmap(norm(c)))
        handles.append(handle)
    plt.close()
    
    for e, maze_name in zip(epochs, maze_names):
        fig, axs = plt.subplots(rows, columns, figsize = (columns * 10, rows * 10))
        fig.suptitle("Epoch {}".format(e), y = 1.05)
        plot_position = (0, 0)
        for arg_name in complete_order:
            if(  arg_name == "break"): plot_position = (plot_position[0] + 1, 0)
            elif(arg_name == "empty_space"): 
                ax = axs[plot_position[0], plot_position[1]] if rows > 1 else axs[plot_position[1]]
                ax.axis("off")
                plot_position = (plot_position[0], plot_position[1] + 1)
            else:
                for plot_dict in plot_dicts:
                    if(plot_dict["arg_name"] == arg_name): break

                ax = axs[plot_position[0], plot_position[1]] if rows > 1 else axs[plot_position[1]]
                arena_map = plt.imread("arenas/{}.png".format(maze_name))
                arena_map = np.flip(arena_map, 0)    
                h, w, _ = arena_map.shape
                extent = [-.5, w-.5, -h+.5, .5]
                ax.imshow(arena_map, extent = extent, zorder = 1, origin = "lower") 
                for c, a in enumerate(agents):
                    for ep in range(episodes):
                        path = plot_dict["pos_lists"]["{}_{}".format(a, e)][ep][1:]
                        xs = [p[1] for p in path] ; ys = [-p[0] for p in path]
                        ax.plot(xs, ys, color=cmap(norm(c)))
                        
                ax.set_title("{}".format(plot_dict["arg_name"]))
                ax.axis("off")
            
                plot_position = (plot_position[0], plot_position[1] + 1)
                        
        fig.legend(loc = "upper left", handles = handles, labels= ["Agent {}".format(a) for a in agents])
        buf = BytesIO()
        plt.savefig(buf, format = "png", bbox_inches = "tight")
        buf.seek(0)
        im = imageio.imread(buf)
        images.append(im)
        buf.close()
        plt.close()
            
        print("Done with epoch {}:\t{}.".format(e, duration()), flush = True)
                
    imageio.mimwrite("saved/hard_video.mp4", images, fps = 1/3)
    
    

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

if(easy): print("\nPlotting positions in easy maze.\n")    ; easy_plotting_pos(complete_easy_order, easy_plot_dicts)
if(hard): print("\nPlotting positions in hard maze(s).\n") ; hard_plotting_pos(complete_hard_order, hard_plot_dicts)   

print("\nDuration: {}. Done!".format(duration()), flush = True)