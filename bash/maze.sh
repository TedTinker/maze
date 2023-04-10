#!/bin/bash -l

eval $1
eval $2
eval $3

real_arg_list=()

for arg in ${arg_list[*]}
do
    temp_file=$(mktemp)
    singularity exec maze.sif python -c "from maze.bash.slurmcraft import all_like_this; result = all_like_this('$arg'); print(result, file=open('${temp_file}', 'w'))"
    returned_value=$(cat ${temp_file})
    real_arg_list=$(echo "${real_arg_list}${returned_value}" | jq -s 'add')
    rm ${temp_file}
done

arg_list=$real_arg_list
singularity exec maze.sif python maze/bash/slurmcraft.py --comp ${comp} --agents ${agents} --arg_list "${arg_list}"
arg_list=$(echo "${arg_list}" | tr -d '[]"' | sed 's/,/, /g')
wait

max_agents=36
jid_list=()
echo
for arg in ${arg_list//, / }
do
    previous_agents=0 
    if [ $arg == "break" ]
    then
        :
    elif [ $arg == "empty_space" ]
    then
        :
    elif [ ${agents} -gt ${max_agents} ]
    then
        num_jobs=$(( ${agents} / ${max_agents} )) 
        remainder=$(( ${agents} % ${max_agents} ))
        if [ $remainder -gt 0 ]
        then
            num_jobs=$(( num_jobs + 1 ))
        else 
            remainder=${max_agents}
        fi
        for (( i=1; i<=${num_jobs}; i++ ))
        do
            if [ $i -eq ${num_jobs} ]
            then
                agents_per_job=$(( remainder ))
            else
                agents_per_job=$(( ${max_agents} ))
            fi
            jid=$(sbatch --export=agents_per_job=${agents_per_job},previous_agents=${previous_agents} maze/bash/main_${arg}.slurm | awk '{print $4}')
            echo "$arg (part $i): $jid"
            jid_list+=($jid)
            previous_agents=$(( previous_agents + agents_per_job ))
        done
    else
        jid=$(sbatch --export=agents_per_job=${agents},previous_agents=0 maze/bash/main_${arg}.slurm | awk '{print $4}')
        echo "$arg: $jid"
        jid_list+=($jid)
    fi
done

job_ids=$(echo ${jid_list[@]} | tr ' ' ':')  
jid=$(sbatch --dependency=afterok:${job_ids} maze/bash/plotting.slurm | awk '{print $4}')
echo
echo "plotting: $jid"
echo