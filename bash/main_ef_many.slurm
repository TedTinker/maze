#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 48:00:00
#SBATCH --mem=50G

#SBATCH --ntasks=36
module load singularity
singularity exec maze.sif python maze/main.py --comp deigo --arg_name ef_many --alpha None --curiosity free --beta 0.01 --hard_maze True --maze_list "['1', '2', '3']" --image_size 8 --max_steps 30 --min_speed 0 --max_speed 150 --naive_eta 2 --free_eta 1 --agents_per_pos_list 36 --epochs "[500, 2000, 4000]" --default_reward "[(1,0)]" --better_reward "[(1,30)]" --wall_punishment -1 --step_lim_punishment 0 --target_entropy 0  --agents $agents_per_job --previous_agents $previous_agents
