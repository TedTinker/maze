import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import imageio
from io import BytesIO
import os
import numpy as np
from skimage.transform import resize

from utils import args, duration, load_dicts, print
from easy_maze import Easy_Maze

print("name:\n{}".format(args.arg_name))



def easy_plotting_pos(complete_order, plot_dicts):
    images = []
    rows = 0 ; columns = 0 ; current_count = 0 
    for arg_name in complete_order: 
        if(arg_name == "break"): rows += 1 ; columns = max(columns, current_count) ; current_count = 0
        else: current_count += 1
    columns = max(columns, current_count)
    if(complete_order[-1] != "break"): rows += 1
        
    epochs_maze_names = list(set(["_".join(key.split("_")[1:]) for key in plot_dicts[0]["pos_lists"].keys()]))
    epochs_maze_names.sort(key=lambda x: (int(x.split('_')[0]), x.split('_')[1]))
    first_arena_name = plot_dicts[0]["args"].maze_list[0] 
    steps = len(plot_dicts[0]["pos_lists"]["1_0_{}".format(first_arena_name)][0])
    agents = list(set([int(key.split("_")[0]) for key in plot_dicts[0]["pos_lists"].keys()])) ; agents.sort()
    episodes = len(plot_dicts[0]["pred_lists"]["1_0_{}".format(first_arena_name)])
    
    cmap = plt.cm.get_cmap("gray_r")
    norm = Normalize(vmin = 0, vmax = 1)
    handles = []
    for c in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:
        handle = plt.scatter(0, 0, marker = "s", s = 250, facecolor = cmap(norm(c)))
        handles.append(handle)
    plt.close()
    
    for epoch_maze_name in epochs_maze_names:
        epoch, maze_name = epoch_maze_name.split("_")
        for step in range(steps):
            fig, axs = plt.subplots(rows, columns, figsize = (columns * 3, rows * 1.5))
            fig.suptitle("Epoch {} (Maze {}) Step {}".format(epoch, maze_name, step), y = 1.1)
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
                    positions = [spot.pos for spot in Easy_Maze(maze_name, plot_dict["args"]).maze.spots]
                    for spot in positions:
                        ax.text(spot[0], spot[1], "\u25A1", fontsize = 30, ha = "center", va = "center")
                    to_plot = {}
                    for agent in agents:
                        for episode in range(episodes):
                            coordinate = plot_dict["pos_lists"]["{}_{}_{}".format(agent, epoch, maze_name)][episode][step]
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
            
        print("{}:\tDone with epoch {} (maze {}).".format(duration(), epoch, maze_name))
            
    x_max = None ; y_max = None 
    for image in images:
        x, y = image.shape[0], image.shape[1]
        x_max = x if x_max == None else max([x_max, x])
        y_max = y if y_max == None else max([y_max, y])
    resized = []
    for image in images: resized.append(resize(image, (x_max, y_max, 4)))
    imageio.mimwrite("easy_video.mp4", resized, fps = 3)
            
            
        
def hard_plotting_pos(complete_order, plot_dicts):
    os.chdir("..")
    images = []
    rows = 0 ; columns = 0 ; current_count = 0 
    for arg_name in complete_order: 
        if(arg_name == "break"): rows += 1 ; columns = max(columns, current_count) ; current_count = 0
        else: current_count += 1
    columns = max(columns, current_count)
    if(complete_order[-1] != "break"): rows += 1
    
    epochs_maze_names = list(set(["_".join(key.split("_")[1:]) for key in plot_dicts[0]["pos_lists"].keys()]))
    epochs_maze_names.sort(key=lambda x: (int(x.split('_')[0]), x.split('_')[1]))
    agents = list(set([int(key.split("_")[0]) for key in plot_dicts[0]["pos_lists"].keys()])) ; agents.sort()
    first_arena_name = plot_dicts[0]["args"].maze_list[0] 
    episodes = len(plot_dicts[0]["pos_lists"]["1_0_{}".format(first_arena_name)])
            
    cmap = plt.cm.get_cmap("hsv", len(agents))
    norm = Normalize(vmin = 0, vmax = len(agents))
    handles = []
    for c in agents:
        handle = plt.scatter(c, 0, marker = "s", s = 10, color=cmap(norm(c)))
        handles.append(handle)
    plt.close()
    
    for epoch_maze_name in epochs_maze_names:
        epoch, maze_name = epoch_maze_name.split("_")
        fig, axs = plt.subplots(rows, columns, figsize = (columns * 10, rows * 10))
        fig.suptitle("Epoch {} (Maze {})".format(epoch, maze_name), y = 1.05)
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
                for c, agent in enumerate(agents):
                    for episode in range(episodes):
                        path = plot_dict["pos_lists"]["{}_{}_{}".format(agent, epoch, maze_name)][episode][1:]
                        xs = [p[1] for p in path] ; ys = [-p[0] for p in path]
                        ax.plot(xs, ys, color=cmap(norm(c)))
                        
                ax.set_title("{}".format(plot_dict["arg_name"]))
                ax.axis("off")
            
                plot_position = (plot_position[0], plot_position[1] + 1)
                        
        fig.legend(loc = "upper left", handles = handles, labels= ["Agent {}".format(agent) for agent in agents])
        buf = BytesIO()
        plt.savefig(buf, format = "png", bbox_inches = "tight")
        buf.seek(0)
        im = imageio.imread(buf)
        print(epoch, maze_name, im.shape)
        images.append(im)
        buf.close()
        plt.close()
            
        print("{}:\tDone with epoch {} (maze {}).".format(duration(), epoch, maze_name))
    
    x_max = None ; y_max = None 
    for image in images:
        x, y = image.shape[0], image.shape[1]
        x_max = x if x_max == None else max([x_max, x])
        y_max = y if y_max == None else max([y_max, y])
    resized = []
    for image in images: resized.append(resize(image, (x_max, y_max, 4)))
    imageio.mimwrite("saved/hard_video.mp4", resized, fps = 1/3)
    
    

plot_dicts, min_max_dict, (easy, complete_easy_order, easy_plot_dicts), (hard, complete_hard_order, hard_plot_dicts) = load_dicts(args)               
if(easy): print("\nPlotting positions in easy maze.\n")    ; easy_plotting_pos(complete_easy_order, easy_plot_dicts)
if(hard): print("\nPlotting positions in hard maze(s).\n") ; hard_plotting_pos(complete_hard_order, hard_plot_dicts)   
print("\nDuration: {}. Done!".format(duration()))