import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal

import matplotlib.pyplot as plt 
import numpy as np
import datetime 

def duration(start_time):
    change_time = datetime.datetime.now() - start_time
    change_time = change_time - datetime.timedelta(microseconds=change_time.microseconds)
    return(change_time)



def get_quantiles(plot_dict, name):
    xs = [i for i, x in enumerate(plot_dict[name][0]) if x != None]
    lists = np.array(plot_dict[name], dtype=float)    
    lists = lists[:,xs]
    quantile_dict = {"xs" : xs}
    quantile_dict["min"] = np.min(lists, 0)
    quantile_dict["q10"] = np.quantile(lists, .1, 0)
    quantile_dict["q20"] = np.quantile(lists, .2, 0)
    quantile_dict["q30"] = np.quantile(lists, .3, 0)
    quantile_dict["q40"] = np.quantile(lists, .4, 0)
    quantile_dict["med"] = np.quantile(lists, .5, 0)
    quantile_dict["q60"] = np.quantile(lists, .6, 0)
    quantile_dict["q70"] = np.quantile(lists, .7, 0)
    quantile_dict["q80"] = np.quantile(lists, .8, 0)
    quantile_dict["q90"] = np.quantile(lists, .9, 0)
    quantile_dict["max"] = np.max(lists, 0)
    return(quantile_dict)



def awesome_plot(here, quantile_dict, color, label, min_max = None, line_transparency = .9, fill_transparency = .1):
    here.fill_between(quantile_dict["xs"], quantile_dict["min"], quantile_dict["max"], color = color, alpha = fill_transparency, linewidth = 0)
    here.fill_between(quantile_dict["xs"], quantile_dict["q10"], quantile_dict["q90"], color = color, alpha = fill_transparency, linewidth = 0)    
    here.fill_between(quantile_dict["xs"], quantile_dict["q20"], quantile_dict["q80"], color = color, alpha = fill_transparency, linewidth = 0)
    here.fill_between(quantile_dict["xs"], quantile_dict["q30"], quantile_dict["q70"], color = color, alpha = fill_transparency, linewidth = 0)
    here.fill_between(quantile_dict["xs"], quantile_dict["q40"], quantile_dict["q60"], color = color, alpha = fill_transparency, linewidth = 0)
    handle, = here.plot(quantile_dict["xs"], quantile_dict["med"], color = color, alpha = line_transparency, label = label)
    if(min_max != None and min_max[0] != min_max[1]): here.set_ylim([min_max[0], min_max[1]])
    return(handle)
    
    
    
def many_min_max(min_max_list):
    mins = [min_max[0] for min_max in min_max_list]
    maxs = [min_max[1] for min_max in min_max_list]
    return((min(mins), max(maxs)))



def plots(plot_dicts, min_max_dict):
    start_time = datetime.datetime.now()
    fig, axs = plt.subplots(13, len(plot_dicts), figsize = (10*len(plot_dicts), 75))
                
    for i, plot_dict in enumerate(plot_dicts):
    
        # Cumulative rewards
        rew_dict = get_quantiles(plot_dict, "rewards")
        max_rewards = [10*i for i in range(len(plot_dict["rewards"][0]))]
        min_rewards = [-1*i for i in range(len(plot_dict["rewards"][0]))]
        
        ax = axs[0,i] if len(plot_dicts) > 1 else axs[0]
        awesome_plot(ax, rew_dict, "turquoise", "Reward")
        ax.axhline(y = 0, color = 'black', linestyle = '--', alpha = .2)
        ax.set_ylabel("Reward")
        ax.set_title(plot_dict["title"] + "\nCumulative Rewards")
        
        ax = axs[1,i] if len(plot_dicts) > 1 else axs[1]
        awesome_plot(ax, rew_dict, "turquoise", "Reward", min_max_dict["rewards"])
        ax.axhline(y = 0, color = "black", linestyle = '--', alpha = .2)
        ax.plot(max_rewards, color = "black", label = "Max Reward")
        ax.plot(min_rewards, color = "black", label = "Max Reward")
        ax.set_ylabel("Reward")
        ax.set_title(plot_dict["title"] + "\nCumulative Rewards, shared min/max")
    
    
    
        # Ending spot
        ax = axs[2,i] if len(plot_dicts) > 1 else axs[2]
        kinds = ["NONE", "BAD", "GOOD"]
        per = len(kinds) * len(plot_dict["spot_names"])
        which = 0
        for j, kind in enumerate(kinds):
            for spot_names in plot_dict["spot_names"]:
                ax.scatter([0], [which], color = (0,0,0,0))
                is_kind = [k for k, spot_name in enumerate(spot_names) if spot_name == kind]
                ax.scatter(is_kind, [which for _ in is_kind], color = "gray", alpha = .1)
                which += 1
            if(j != len(kinds)-1):
                ax.plot([x for x in range(len(spot_names))], [which - .5 for _ in range(len(spot_names))], color = "black")
        y_space = which/len(kinds)/2
        ax.set_yticks([y_space + 2*y_space*j for j in range(len(kinds))], kinds, rotation='vertical')
        ax.set_ylabel("Ending Spot")
        ax.set_title(plot_dict["title"] + "\nEnding Spots")
        
        
        
        # Losses
        mse_dict = get_quantiles(plot_dict, "mse")
        dkl_dict = get_quantiles(plot_dict, "dkl")
        alpha_dict = get_quantiles(plot_dict, "alpha")
        actor_dict = get_quantiles(plot_dict, "actor")
        crit1_dict = get_quantiles(plot_dict, "critic_1")
        crit2_dict = get_quantiles(plot_dict, "critic_2")
        
        ax = axs[3,i] if len(plot_dicts) > 1 else axs[3]
        h1 = awesome_plot(ax, mse_dict, "green", "MSE")
        ax.set_ylabel("MSE Loss")
        ax2 = ax.twinx()
        h2 = awesome_plot(ax2, dkl_dict, "red", "DKL")
        ax2.set_ylabel("DKL Loss")
        ax.legend(handles = [h1, h2])
        ax.set_title(plot_dict["title"] + "\nForward Losses")
        
        ax = axs[4,i] if len(plot_dicts) > 1 else axs[4]
        h1 = awesome_plot(ax, mse_dict, "green", "MSE", min_max_dict["mse"])
        ax.set_ylabel("MSE Loss")
        ax2 = ax.twinx()
        h2 = awesome_plot(ax2, dkl_dict, "red", "DKL", min_max_dict["dkl"])
        ax2.set_ylabel("DKL Loss")
        ax.legend(handles = [h1, h2])
        ax.set_title(plot_dict["title"] + "\nForward Losses, shared min/max")
        
        ax = axs[5,i] if len(plot_dicts) > 1 else axs[5]
        h1 = awesome_plot(ax, actor_dict, "red", "Actor")
        ax.set_ylabel("Actor Loss")
        ax2 = ax.twinx()
        h2 = awesome_plot(ax2, crit1_dict, "blue", "Critic")
        awesome_plot(ax2, crit2_dict, "blue", "Critic")
        ax2.set_ylabel("Critic Losses")
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.08))
        h3 = awesome_plot(ax3, alpha_dict, "black", "Alpha")
        ax3.set_ylabel("Alpha Loss")
        ax.legend(handles = [h1, h2, h3])
        ax.set_title(plot_dict["title"] + "\nOther Losses")
        
        ax = axs[6,i] if len(plot_dicts) > 1 else axs[6]
        min_max = many_min_max([min_max_dict["critic_1"], min_max_dict["critic_2"]])
        h1 = awesome_plot(ax, actor_dict, "red", "Actor", min_max_dict["actor"])
        ax.set_ylabel("Actor Loss")
        ax2 = ax.twinx()
        h2 = awesome_plot(ax2, crit1_dict, "blue", "Critic", min_max)
        awesome_plot(ax2, crit2_dict, "blue", "Critic", min_max)
        ax2.set_ylabel("Critic Losses")
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.08))
        h3 = awesome_plot(ax3, alpha_dict, "black", "Alpha", min_max_dict["alpha"])
        ax3.set_ylabel("Alpha Loss")
        ax.legend(handles = [h1, h2, h3])
        ax.set_title(plot_dict["title"] + "\nOther Losses, shared min/max")
        
        
        
        # Extrinsic and Intrinsic rewards
        ext_dict = get_quantiles(plot_dict, "extrinsic")
        cur_dict = get_quantiles(plot_dict, "intrinsic_curiosity")
        ent_dict = get_quantiles(plot_dict, "intrinsic_entropy")
        
        ax = axs[7,i] if len(plot_dicts) > 1 else axs[7]
        handles = []
        handles.append(awesome_plot(ax, ext_dict, "red", "Extrinsic"))
        ax.set_ylabel("Extrinsic")
        if((cur_dict["min"] != cur_dict["max"]).all()):
            ax2 = ax.twinx()
            handles.append(awesome_plot(ax2, cur_dict, "green", "Curiosity"))
            ax2.set_ylabel("Curiosity")
        if((ent_dict["min"] != ent_dict["max"]).all()):
            ax3 = ax.twinx()
            ax3.spines["right"].set_position(("axes", 1.08))
            handles.append(awesome_plot(ax3, ent_dict, "blue", "Entropy"))
            ax3.set_ylabel("Entropy")
        ax.legend(handles = handles)
        ax.set_title(plot_dict["title"] + "\nExtrinsic and Intrinsic Rewards")
        
        ax = axs[8,i] if len(plot_dicts) > 1 else axs[8]
        handles = []
        handles.append(awesome_plot(ax, ext_dict, "red", "Extrinsic", min_max_dict["extrinsic"]))
        ax.set_ylabel("Extrinsic")
        if((cur_dict["min"] != cur_dict["max"]).all()):
            ax2 = ax.twinx()
            handles.append(awesome_plot(ax2, cur_dict, "green", "Curiosity", min_max_dict["intrinsic_curiosity"]))
            ax2.set_ylabel("Curiosity")
        if((ent_dict["min"] != ent_dict["max"]).all()):
            ax3 = ax.twinx()
            ax3.spines["right"].set_position(("axes", 1.08))
            handles.append(awesome_plot(ax3, ent_dict, "blue", "Entropy", min_max_dict["intrinsic_entropy"]))
            ax3.set_ylabel("Entropy")
        ax.legend(handles = handles)
        ax.set_title(plot_dict["title"] + "\nExtrinsic and Intrinsic Rewards, shared min/max")
        
        
        
        # DKL
        dkl_dict = get_quantiles(plot_dict, "dkl_change")
    
        ax = axs[9,i] if len(plot_dicts) > 1 else axs[9]
        awesome_plot(ax, dkl_dict, "green", "DKL")
        ax.set_ylabel("DKL")
        ax.set_title(plot_dict["title"] + "\nDKL")
        
        ax = axs[10,i] if len(plot_dicts) > 1 else axs[10]
        awesome_plot(ax, dkl_dict, "green", "DKL", min_max_dict["dkl_change"])
        ax.set_ylabel("DKL")
        ax.set_title(plot_dict["title"] + "\nDKL, shared min/max")
        
        
        
        # Curiosities
        naive_dict = get_quantiles(plot_dict, "naive")
        friston_dict = get_quantiles(plot_dict, "friston")
        
        ax = axs[11,i] if len(plot_dicts) > 1 else axs[11]
        awesome_plot(ax, naive_dict, "green", "Naive")
        awesome_plot(ax, friston_dict, "red", "Friston")
        ax.legend()
        ax.set_ylabel("Curiosities")
        ax.set_title(plot_dict["title"] + "\nCuriosities")
        
        min_max = many_min_max([min_max_dict["naive"], min_max_dict["friston"]])
        ax = axs[12,i] if len(plot_dicts) > 1 else axs[12]
        awesome_plot(ax, naive_dict, "green", "Naive", min_max)
        awesome_plot(ax, friston_dict, "red", "Friston", min_max)
        ax.legend()
        ax.set_ylabel("Curiosities")
        ax.set_title(plot_dict["title"] + "\nCuriosities, shared min/max")
        
        print(i, plot_dict["title"], duration(start_time))

    
    
    # Done!
    fig.tight_layout(pad=1.0)
    plt.savefig("saved/plot.png", bbox_inches = "tight")
    plt.show()
    plt.close()