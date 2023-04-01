#!/bin/bash -l

eval $1
eval $2
eval $3

real_job_list=()

for job in ${job_list[*]}
do
    temp_file=$(mktemp)
    singularity exec t_maze.sif python -c "from easy_maze.bash.slurmcraft import all_like_this; result = all_like_this('$job'); print(result, file=open('${temp_file}', 'w'))"
    returned_value=$(cat ${temp_file})
    real_job_list=$(echo "${real_job_list}${returned_value}" | jq -s 'add')
    rm ${temp_file}
done

echo
echo -n "Original job list: "
echo "${job_list[*]}" | sed "s/ /, /g"
echo
echo new_job_list: $real_job_list
echo
job_list=$real_job_list

singularity exec t_maze.sif python easy_maze/bash/slurmcraft.py --comp ${comp} --agents ${agents} --arg_list "${job_list}"

job_list=$(echo "${job_list}" | tr -d '[]"' | sed 's/,/, /g')
for job in ${job_list//, / }
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