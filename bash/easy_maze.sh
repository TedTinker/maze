#!/bin/bash -l

eval $1
eval $2
eval $3

real_arg_list=()

for arg in ${arg_list[*]}
do
    temp_file=$(mktemp)
    singularity exec t_maze.sif python -c "from easy_maze.bash.slurmcraft import all_like_this; result = all_like_this('$arg'); print(result, file=open('${temp_file}', 'w'))"
    returned_value=$(cat ${temp_file})
    real_arg_list=$(echo "${real_arg_list}${returned_value}" | jq -s 'add')
    rm ${temp_file}
done

arg_list=$real_arg_list
singularity exec t_maze.sif python easy_maze/bash/slurmcraft.py --comp ${comp} --agents ${agents} --arg_list "${arg_list}"
arg_list=$(echo "${arg_list}" | tr -d '[]"' | sed 's/,/, /g')
wait

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
    elif [ ${agents} -gt 25 ]
    then
        num_jobs=$(( ${agents} / 25 )) 
        remainder=$(( ${agents} % 25 ))
        if [ $remainder -gt 0 ]
        then
            num_jobs=$(( num_jobs + 1 ))
        fi
        for (( i=1; i<=${num_jobs}; i++ ))
        do
            if [ $i -eq ${num_jobs} ]
            then
                agents_per_job=$(( remainder ))
            else
                agents_per_job=$(( 25 ))
            fi
            jid=$(sbatch easy_maze/bash/main_${arg}.slurm --export=agents=${agents_per_job},previous_agents=${previous_agents} | awk '{print $4}')
            echo "$arg (part $i): $jid"
            jid_list+=($jid)
            previous_agents=$(( previous_agents + agents_per_job ))
        done
    else
        jid=$(sbatch easy_maze/bash/main_${arg}.slurm --export=agents=${agents},previous_agents=${previous_agents} | awk '{print $4}')
        echo "$arg: $jid"
        jid_list+=($jid)
        previous_agents=$(( previous_agents + agents )) 
    fi
done

post_jid_list=()
echo
for arg in ${arg_list//, / }
do
    job_ids=$(echo ${jid_list[@]} | tr ' ' ':')  
    jid=$(sbatch --dependency=afterok:${job_ids} easy_maze/bash/post_${arg}.slurm | awk '{print $4}')
    echo "post_$arg: $jid"
    post_jid_list+=($jid)
done

echo
# Run plotting job
dependency_string=$(IFS=:; echo "${post_jid_list[*]}")
jid=$(sbatch --dependency=afterok:${dependency_string} easy_maze/bash/plotting.slurm | awk '{print $4}')
echo "plotting: $jid"
echo