import matplotlib.pyplot as plt
import os

from utils import args, duration, print

print("name:\n{}".format(args.arg_name))

os.chdir("saved/thesis_pics")
files = os.listdir() ; files.sort()
rewards_files = [file for file in files if file.endswith("rewards.png")]
exits_files = [file for file in files if file.endswith("exits.png")]
arg_names = ["_".join(rewards.split("_")[:-1]) for rewards in rewards_files]

paths_files = []
for arg_name in arg_names:
    paths_files.append([file for file in files if file.startswith("{}_paths".format(arg_name))])

for (arg_name, rewards, exits, paths_list) in zip(arg_names, rewards_files, exits_files, paths_files):
    if(len(paths_list) == 1):
        fig, axs = plt.subplots(3, 1)
        axs[0].imshow(plt.imread(rewards))       ; axs[0].axis("off")
        axs[1].imshow(plt.imread(exits))         ; axs[1].axis("off")
        axs[2].imshow(plt.imread(paths_list[0])) ; axs[2].axis("off")
    else:
        fig, axs = plt.subplots(2, max([2, len(paths_list)]))
        axs[0,0].imshow(plt.imread(rewards)) ; axs[0,0].axis("off")
        axs[0,1].imshow(plt.imread(exits))   ; axs[1,0].axis("off")
        for i, paths in enumerate(paths_list):
            axs[1,i].imshow(plt.imread(paths_list[i])); axs[1,i].axis("off")

    fig.tight_layout(pad=1.0)
    plt.savefig("{}.png".format(arg_name), bbox_inches = "tight")
    plt.close(fig)
    
    os.remove(rewards)
    os.remove(exits)
    for paths in paths_list:
        os.remove(paths)

print("\nDuration: {}. Done!".format(duration()))