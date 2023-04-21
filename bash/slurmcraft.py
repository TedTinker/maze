#%%
import argparse, ast, json
parser = argparse.ArgumentParser()
parser.add_argument("--comp",         type=str,  default = "deigo")
parser.add_argument("--agents",       type=int,  default = 10)
parser.add_argument("--arg_list",     type=str,  default = ["d", "e", "en1"])
try:    args = parser.parse_args()
except: args, _ = parser.parse_known_args()

if(type(args.arg_list) != list): args.arg_list = json.loads(args.arg_list)
combined = "___{}___".format("+".join(args.arg_list))    

import os 
try:    os.chdir("maze/bash")
except: pass



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
                nums = [min_val + i*((max_val - min_val) / (num - 1)) for i in range(num)]
                hard_args.append([arg_words[0], nums])
            else: hard_args.append([arg_words[0], arg_words[1:]])    
    if(hard_args == []): combos = [easy_args]
    else:
        combos = list(product(*[args[1] for args in hard_args]))
        combos = [" ".join([hard_args[i][0] + " " + str(combo[i]) for i in range(len(hard_args))]) for combo in combos]
        combos = [easy_args + combo for combo in combos]
    return(combos)

slurm_dict = {
    "d"    : "", 
    "e"    : "--alpha None",
    "n"    : "--curiosity naive",
    "en"   : "--alpha None --curiosity naive",
    "ef"   : "--alpha None --curiosity free"
}

def add_this(name, this):
    keys, values = [], []
    for key, value in slurm_dict.items(): keys.append(key) ; values.append(value)
    for key, value in zip(keys, values):  
        this_this = this
        if(key[-1] == "_"): key = key[:-1] ; this_this += "_"
        slurm_dict[key + "_" + name] = value + " " + this_this  
add_this("hard",      "--hard_maze True --agents_per_pos_list 10 --maze_list \"('t',)\"")
add_this("log_prob", "--accuracy log_prob")
add_this("rand",      "--randomness 10")

new_slurm_dict = {}
for key, item in slurm_dict.items():
    combos = expand_args(item)
    if(len(combos) == 1): new_slurm_dict[key] = combos[0] 
    else:
        for i, combo in enumerate(combos): new_slurm_dict[key + str(i+1)] = combo
        
slurm_dict = new_slurm_dict

def all_like_this(this): 
    if(this in ["break", "empty_space"]): result = [this]
    elif(this[-1] != "_"):                result = [this]
    else: result = [key for key in slurm_dict.keys() if key.startswith(this) and key[len(this):].isdigit()]
    return(json.dumps(result))
            


max_cpus = 36
if(__name__ == "__main__"):
    
    if(args.comp == "deigo"):
        partition = """
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --time 2:00:00
"""

    if(args.comp == "saion"):
        partition = """
#SBATCH --partition=taniu
#SBATCH --gres=gpu:1
#SBATCH --time 48:00:00
"""


            
    for name in args.arg_list:
        if(name in ["break", "empty_space"]): pass 
        else:
            with open("main_{}.slurm".format(name), "w") as f:
                f.write(
"""
#!/bin/bash -l
{}
#SBATCH --ntasks={}
#SBATCH --mem=2G

module load singularity
singularity exec maze.sif python maze/main.py --arg_name {} {} --agents $agents_per_job --previous_agents $previous_agents
""".format(partition, max_cpus, name, slurm_dict[name])[1:])
            


    with open("finish_dicts.slurm", "w") as f:
        f.write(
"""
#!/bin/bash -l
{}
#SBATCH --mem=2G

module load singularity
singularity exec maze.sif python maze/finish_dicts.py --arg_title {} --arg_name finishing_dictionaries
""".format(partition, combined)[1:])
        
    with open("plotting.slurm", "w") as f:
        f.write(
"""
#!/bin/bash -l
{}
#SBATCH --mem=2G

module load singularity
singularity exec maze.sif python maze/plotting.py --arg_title {} --arg_name plotting
""".format(partition, combined)[1:])
        
    with open("plotting_pred.slurm", "w") as f:
        f.write(
"""
#!/bin/bash -l
{}
#SBATCH --mem=2G

module load singularity
singularity exec maze.sif python maze/plotting_pred.py --arg_title {} --arg_name plotting_predictions
""".format(partition, combined)[1:])
        
    with open("plotting_pos.slurm", "w") as f:
        f.write(
"""
#!/bin/bash -l
{}
#SBATCH --mem=2G

module load singularity
singularity exec maze.sif python maze/plotting_pos.py --arg_title {} --arg_name plotting_positions
""".format(partition, combined)[1:])
# %%

