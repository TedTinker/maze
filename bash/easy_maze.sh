#!/bin/bash -l

eval $1
eval $2
eval $3

job_list_arg="['${job_list[*]}']"
job_list_arg=${job_list_arg// /\',\'}
singularity exec t_maze.sif python easy_maze/bash/slurmcraft.py --comp ${comp} --agents ${agents} --arg_list "${job_list_arg}"

jid_list=()

for job in ${job_list[*]}
do
    if [ $job == "break" ]
    then
        :
    elif [ $job == "empty_space" ]
    then
        :
    else
        jid=$(sbatch easy_maze/bash/main_$job.slurm)
        echo $jid
        jid=(${jid// / })
        jid=${jid[3]}     
        jid_list+=($jid)
    fi
done

jid=$(sbatch --dependency afterok:$(echo ${jid_list[*]} | tr ' ' :) easy_maze/bash/post_final.slurm)
echo $jid