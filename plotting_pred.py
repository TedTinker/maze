import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import torch
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
                        rows = len(pred_list) ; columns = 3 + 2 * plot_dict["args"].samples_per_pred
                        fig, axs = plt.subplots(rows, columns, figsize = (columns * 3, rows * 1.5))
                        title = "Agent {}: Epoch {}, Episode {}".format(agent, epoch, episode)
                        fig.suptitle(title, y = 1.1)
                        for row, (obs, zp_mu_pred, zp_preds, zq_mu_pred, zq_preds) in enumerate(pred_list):
                            for column in range(columns):
                                ax = axs[row, column] ; ax.axis("off")
                                if(row == 0 and column > 0): pass
                                else:                
                                    # Actual obs
                                    if(column == 0):   
                                        ax.scatter([x for x in range(obs_size)], [0 for _ in range(obs_size)], marker = "s", s = 250, linewidths = 1, edgecolor='blue', cmap = cmap, c = obs, norm = norm)
                                        ax.set_title("Step {}".format(row))
                                    # ZP Mean
                                    elif(column == 1): 
                                        ax.scatter([x for x in range(obs_size)], [0 for _ in range(obs_size)], marker = "s", s = 250, linewidths = 1, edgecolor='blue', cmap = cmap, c = zp_mu_pred, norm = norm)
                                        ax.set_title("ZP Mean")
                                    # ZP Samples
                                    elif(column in [i+2 for i in range(plot_dict["args"].samples_per_pred)]):
                                        pred_num = column - 2
                                        pred = zp_preds[pred_num]
                                        ax.scatter([x for x in range(obs_size)], [0 for _ in range(obs_size)], marker = "s", s = 250, linewidths = 1, edgecolor='blue', cmap = cmap, c = pred, norm = norm)
                                        ax.set_title("ZP Sample {}".format(pred_num+1))
                                    # ZQ Mean
                                    elif(column == 2 + plot_dict["args"].samples_per_pred):
                                        ax.scatter([x for x in range(obs_size)], [0 for _ in range(obs_size)], marker = "s", s = 250, linewidths = 1, edgecolor='blue', cmap = cmap, c = zq_mu_pred, norm = norm)
                                        ax.set_title("ZQ Mean")
                                    # ZQ Samples
                                    else:
                                        pred_num = column - 3 - plot_dict["args"].samples_per_pred
                                        pred = zq_preds[pred_num]
                                        ax.scatter([x for x in range(obs_size)], [0 for _ in range(obs_size)], marker = "s", s = 250, linewidths = 1, edgecolor='blue', cmap = cmap, c = pred, norm = norm)
                                        ax.set_title("ZQ Sample {}".format(pred_num+1))
                        plt.savefig("{}/{}.png".format(arg_name, title), format = "png", bbox_inches = "tight")
                        plt.close()
                        
        print("{}:\tDone with epoch {}.".format(duration(), epoch), flush = True)
                                
                                                
        
def hard_plotting_pred(complete_order, plot_dicts):
    epochs = list(set([int(key.split("_")[1]) for key in plot_dicts[0]["pred_lists"].keys()])) ; epochs.sort()
    agents = list(set([int(key.split("_")[0]) for key in plot_dicts[0]["pred_lists"].keys()])) ; agents.sort()
    episodes = len(plot_dicts[0]["pred_lists"]["0_0"])
        
    for epoch in epochs:
        for agent in agents:
            for arg_name in complete_order:
                if(arg_name in ["break", "empty_space"]): pass 
                else:
                    for plot_dict in plot_dicts:
                        if(plot_dict["arg_name"] == arg_name): pred_lists = plot_dict["pred_lists"]["{}_{}".format(agent, epoch)] ; break 
                    for episode in range(episodes):
                        pred_list = pred_lists[episode]
                        rows = len(pred_list) ; columns = 2 + plot_dict["args"].samples_per_pred
                        fig, axs = plt.subplots(rows, columns, figsize = (columns * 2, rows * 1.5))
                        title = "Agent {}: Epoch {}, Episode {}".format(agent, epoch, episode)
                        fig.suptitle(title, y = 1.1)
                        for row, ((rgbd, spe), ((rgbd_mu_pred_p, pred_rgbd_p), (spe_mu_pred_p, pred_spe_p)), ((rgbd_mu_pred_q, pred_rgbd_q), (spe_mu_pred_q, pred_spe_q))) in enumerate(pred_list):
                            print(rgbd.shape, spe.shape, rgbd_mu_pred_p.shape, spe_mu_pred_p.shape, flush = True)
                            for column in range(columns):
                                ax = axs[row, column] ; ax.axis("off")
                                if(row == 0 and column > 0): pass
                                else:                
                                    # Actual obs
                                    if(column == 0):   
                                        ax.imshow(rgbd[:,:,0:3])
                                        ax.set_title("Step {}".format(row))
                                    # ZP Mean
                                    elif(column == 1): 
                                        ax.imshow(rgbd_mu_pred_p) # Still gotta add speeds in titles!
                                        ax.set_title("ZP Mean")
                                    # ZP Samples
                                    elif(column in [i+2 for i in range(plot_dict["args"].samples_per_pred)]):
                                        pred_num = column - 2
                                        pred = pred_rgbd_p[pred_num]
                                        ax.imshow(pred)
                                        ax.set_title("ZP Sample {}".format(pred_num+1))
                                    # ZQ Mean
                                    elif(column == 2 + plot_dict["args"].samples_per_pred):
                                        ax.imshow(rgbd_mu_pred_q)
                                        ax.set_title("ZQ Mean")
                                    # ZQ Samples
                                    else:
                                        pred_num = column - 3 - plot_dict["args"].samples_per_pred
                                        pred = pred_rgbd_q[pred_num]
                                        ax.imshow(pred)
                                        ax.set_title("ZQ Sample {}".format(pred_num+1))
                        plt.savefig("{}/{}.png".format(arg_name, title), format = "png", bbox_inches = "tight")
                        plt.close()
                        
        print("{}:\tDone with epoch {}.".format(duration(), epoch), flush = True)
    


plot_dicts, min_max_dict, (easy, complete_easy_order, easy_plot_dicts), (hard, complete_hard_order, hard_plot_dicts) = load_dicts(args)    
if(easy): print("\nPlotting predictions in easy maze.\n")    ; easy_plotting_pred(complete_easy_order, easy_plot_dicts)
if(hard): print("\nPlotting predictions in hard maze(s).\n") ; hard_plotting_pred(complete_hard_order, hard_plot_dicts)    
print("\nDuration: {}. Done!".format(duration()), flush = True)