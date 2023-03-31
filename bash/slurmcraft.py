#%%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--comp",         type=str,  default = "deigo")
parser.add_argument("--arg_title",    type=str,  default = "default")
parser.add_argument("--post",         type=str,  default = "False")
try:    args = parser.parse_args()
except: args, _ = parser.parse_known_args()

import os 
try:    os.chdir("easy_maze/bash")
except: pass

if(args.comp == "deigo"):
	partition = """
#SBATCH --partition=short
#SBATCH --time 2:00:00
"""

if(args.comp == "saion"):
	partition = """
#SBATCH --partition=taniu
#SBATCH --gres=gpu:1
#SBATCH --time 48:00:00
"""



from itertools import product
def expand_args(args = ""):
    if(args == ""): combos = []
    easy_args = ""
    hard_args = []
    arg_split = ["--" + arg for arg in args.split("--") if arg.replace(" ", "") != ""]
    for arg in arg_split:
        arg_words = [word for word in arg.split(" ") if word.replace(" ", "") != ""]
        if(len(arg_words) == 2): easy_args += arg + " "
        else:
            if arg_words[1] == "num_min_max":
                num, min_val, max_val = arg_words[2:]
                num = int(num)
                min_val = float(min_val)
                max_val = float(max_val)
                nums = [min_val + (max_val - min_val) * (n / (num - 1)) for n in range(num)]
                hard_args.append([arg_words[0], nums])
            else: hard_args.append([arg_words[0], arg_words[1:]])    
    if(hard_args == []): combos = [easy_args]
    else:
        combos = list(product(*[args[1] for args in hard_args]))
        combos = [" ".join([hard_args[i][0] + " " + str(combo[i]) for i in range(len(hard_args))]) for combo in combos]
        combos = [easy_args + combo for combo in combos]
    return(combos)

slurm_dict = {
    "d"   : "", 
    "e"   : "--alpha None",

    "n"  : "--curiosity naive_1",
    "en_" : "--alpha None --curiosity naive_1 naive_2 naive_3",

    "f"   : "--curiosity free",
    "ef"  : "--alpha None --curiosity free",
    "ef_" : "--alpha None --curiosity free --free_eta .001 .01 .1"}

new_slurm_dict = {}

for key, item in slurm_dict.items():
    combos = expand_args(item)
    if(len(combos) == 1): new_slurm_dict[key] = combos[0] 
    else:
        for i, combo in enumerate(combos): new_slurm_dict[key + str(i+1)] = combo
        
slurm_dict = new_slurm_dict
            

        
if(args.post == "False"):
    with open("main_{}.slurm".format(args.arg_title), "a") as f:
        f.write(
"""
#!/bin/bash -l
{}
#SBATCH --mem=2G
##SBATCH --constraint 32

module load singularity
singularity exec t_maze.sif python easy_maze/main.py --id ${{SLURM_ARRAY_TASK_ID}} --arg_title {} {}
    """.format(partition, args.arg_title, slurm_dict[args.arg_title])[1:])
        
if(args.post == "True"):
    if(args.arg_title[:3] == "___"): 
        slurm_dict[args.arg_title] = "--name {}".format(args.arg_title)
        name = "final"
    else: name = args.arg_title
    with open("post_{}.slurm".format(name), "a") as f:
        f.write(
"""
#!/bin/bash -l
{}
#SBATCH --mem=2G
##SBATCH --constraint 32

module load singularity
singularity exec t_maze.sif python easy_maze/post_main.py --arg_title {} {}
""".format(partition, args.arg_title, slurm_dict[args.arg_title])[1:])
# %%

