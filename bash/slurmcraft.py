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
try:    os.chdir("easy_maze/bash")
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
    "en_rand"   : "--alpha None --curiosity naive --randomness True",
    
    "en_log_prob" : "--alpha None --curiosity naive --beta_obs 2.5 --beta_zq 0 --accuracy log_prob --naive_eta .07",
    
    "ef"  : "--alpha None --curiosity free --beta_obs 2 --beta_zq 0 --free_eta_obs 7.5",
    "ef_rand"  : "--alpha None --curiosity free --beta_obs 2 --beta_zq 0 --free_eta_obs 7.5 --randomness True",
    
    "ef_log_prob" : "--alpha None --curiosity free --beta_obs 2 --beta_zq 0 --accuracy log_prob --free_eta_obs 7.5",
    
    "d_state" : "--state_forward True",
    
    "e_state"    : "--state_forward True --alpha None",

    "n_state"    : "--state_forward True --curiosity naive",
    "en_state"   : "--state_forward True --alpha None --curiosity naive",
    
    "en_log_prob_state" : "--state_forward True --alpha None --curiosity naive --beta_obs 2.5 --beta_zq 0 --accuracy log_prob --naive_eta .07",
    
    "ef_state"  : "--state_forward True --alpha None --curiosity free --beta_obs 2 --beta_zq 0 --free_eta_obs 7.5",
    
    "ef_log_prob_state" : "--state_forward True --alpha None --curiosity free --beta_obs 2 --beta_zq 0 --accuracy log_prob --free_eta_obs 7.5",
    
    }

new_slurm_dict = {}
for key, item in slurm_dict.items():
    combos = expand_args(item)
    if(len(combos) == 1): new_slurm_dict[key] = combos[0] 
    else:
        for i, combo in enumerate(combos): new_slurm_dict[key + str(i+1)] = combo
        
slurm_dict = new_slurm_dict

def all_like_this(this): 
    if(this[-1] != "_"): result = [this]
    else: result = [key for key in slurm_dict.keys() if key.startswith(this)]
    return(json.dumps(result))
            


max_cpus = 36
if(__name__ == "__main__"):
    
    if(args.comp == "deigo"):
        partition = """
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --time 20:00
"""

    if(args.comp == "saion"):
        partition = """
#SBATCH --partition=taniu
#SBATCH --gres=gpu:1
#SBATCH --time 48:00:00
"""


            
    for name in args.arg_list:
        with open("main_{}.slurm".format(name), "w") as f:
            f.write(
"""
#!/bin/bash -l
{}
#SBATCH --ntasks={}
#SBATCH --mem=2G

module load singularity
singularity exec t_maze.sif python easy_maze/main.py --arg_name {} {} --agents $agents_per_job --previous_agents $previous_agents
""".format(partition, max_cpus, name, slurm_dict[name])[1:])
            

            
    with open("plotting.slurm", "w") as f:
        f.write(
"""
#!/bin/bash -l
{}
#SBATCH --mem=2G

module load singularity
singularity exec t_maze.sif python easy_maze/plotting.py --arg_name {}
""".format(partition, combined)[1:])
# %%

