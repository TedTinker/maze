#%%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name",         type=str,  default = "no_entropy_no_curiosity")
parser.add_argument("--explore_type", type=str,  default = "no_entropy_no_curiosity")
parser.add_argument("--post",         type=str,  default = "False")
try:    args = parser.parse_args()
except: args, _ = parser.parse_known_args()

import os 
try:    os.chdir("easy_maze/bash/saion")
except: pass

if(args.post == "False"):
    slurm_dict = {}
    f = open("slurms.txt", "r")
    slurms = f.readlines()
    for line in slurms:
        if(line == "\n"): pass 
        else:
            name, text = line.split(":")
            slurm_dict[name.strip()] = text.strip()
            
    with open("main_{}.slurm".format(args.name), "a") as f:
        f.write(
"""
#!/bin/bash -l
#SBATCH --partition=taniu
#SBATCH --gres=gpu:1
#SBATCH --time 48:00:00
#SBATCH --mem=32G
##SBATCH --constraint 32

module load singularity
singularity exec t_maze.sif python easy_maze/main.py --id ${{SLURM_ARRAY_TASK_ID}} --explore_type {} {}
    """.format(args.name, slurm_dict[args.name])[1:])
        
if(args.post == "True"):
    with open("post_{}.slurm".format(args.name), "a") as f:
        f.write(
"""
#!/bin/bash -l
#SBATCH --partition=taniu
#SBATCH --gres=gpu:1
#SBATCH --time 48:00:00
#SBATCH --mem=32G
##SBATCH --constraint 32

module load singularity
singularity exec t_maze.sif python easy_maze/post_main.py --explore_type {}
""".format(args.explore_type)[1:])
# %%

