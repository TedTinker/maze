import matplotlib.pyplot as plt
import os
import re

from utils import args, duration, print

print("name:\n{}".format(args.arg_name))

os.chdir("saved/thesis_pics")
files = os.listdir() ; files.sort()
rewards_files = [file for file in files if file.startswith("rewards")]
exits_files = [file for file in files if file.startswith("exits")]
arg_names = ["_".join(rewards.split("_")[1:])[:-4] for rewards in rewards_files]

paths_files = []
for arg_name in arg_names:
    paths_files.append([file for file in files if re.match(r"paths_{}_\d+.png".format(re.escape(arg_name)), file)])
    
real_names = {
    "d"  : "No Entropy, No Curiosity",
    "e"  : "Entropy",
    "n"  : "Naive Curiosity",
    "en" : "Entropy and Naive Curiosity",
    "f"  : "Aware Curiosity",
    "ef" : "Entropy and Aware Curiosity",
}

def add_this(name):
    keys, values = [], []
    for key, value in real_names.items(): keys.append(key) ; values.append(value)
    for key, value in zip(keys, values):  
        new_key = key + "_" + name 
        real_names[new_key] = value
add_this("hard")
add_this("many")

for (arg_name, rewards, exits, paths_list) in zip(arg_names, rewards_files, exits_files, paths_files):
    if(len(paths_list) == 1):
        fig, axs = plt.subplots(3, 1, figsize = (10, 30))
        axs[0].imshow(plt.imread(rewards))       ; axs[0].axis("off")
        axs[1].imshow(plt.imread(exits))         ; axs[1].axis("off")
        axs[2].imshow(plt.imread(paths_list[0])) ; axs[2].axis("off")
    else:
        rows = max([2, len(paths_list)])
        columns = 2
        fig, axs = plt.subplots(rows, columns, figsize = (10 * columns, 10 * rows))
        axs[0,0].imshow(plt.imread(rewards))
        axs[1,0].imshow(plt.imread(exits))   
        for i, paths in enumerate(paths_list):
            axs[i,1].imshow(plt.imread(paths_list[i]))
        for row in range(rows):
            for column in range(columns):
                axs[row, column].axis("off")

    if(arg_name in real_names.keys()): title = real_names[arg_name]
    elif(arg_name.endswith("rand")):   title = "with Curiosity Trap"
    else:                              title = arg_name
    fig.suptitle(title, fontsize=30, y=1.0)
    fig.tight_layout(pad=1.0)
    plt.savefig("{}.png".format(arg_name), bbox_inches = "tight", dpi=100)
    plt.close(fig)
    
    os.remove(rewards)
    os.remove(exits)
    for paths in paths_list:
        os.remove(paths)

print("\nDuration: {}. Done!".format(duration()))