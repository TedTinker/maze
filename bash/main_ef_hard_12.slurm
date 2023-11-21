#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 48:00:00
#SBATCH --mem=50G

#SBATCH --ntasks=36
module load singularity
singularity exec maze.sif python maze/main.py --comp deigo --arg_name ef_hard_12 --alpha None --curiosity free --beta 0.001 --hard_maze True --maze_list "['t']" --epochs "[500]" --image_size 8 --boxes_high 1 --max_steps 30 --min_speed 0 --max_speed 75 --better_reward "[(1,0),(1,10)]" --step_lim_punishment -2 --step_cost 0.99 --state_size 32 --free_eta 1 --target_entropy -3 --agents_per_pos_list 36  --agents $agents_per_job --previous_agents $previous_agents
