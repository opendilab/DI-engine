work_path=$(dirname $0)
#RES_PREFIX=("" "" "" "" "")  # for local test
#RES_PREFIX=("" "" "srun -p $1 --job-name=league " "srun -p $1 --gres=gpu:1 --job-name=learner " "srun -p $1 --gres=gpu:8 --job-name=actor ")  # for slurm lustre single gpu
RES_PREFIX=("" "" "srun -p $1 --job-name=league " "srun --mpi=pmi2 -p $1 -n 2 --gres=gpu:2 --job-name=learner " "srun -p $1 --gres=gpu:8 --job-name=actor ")  # for slurm lustre multi gpu
CMD=("python3 -u -m nervex.system.coordinator_start" "python3 -u -m nervex.system.manager_start" \
     "python3 -u -m nervex.system.league_manager_start" "python3 -u -m nervex.system.learner_start" \
     "python3 -u -m nervex.system.actor_start")
CONFIG=" --config $work_path/sumo_dqn_dist_default_config.yaml"

for ((i=0;i<${#CMD[@]};i++))
do
    ${RES_PREFIX[$i]}${CMD[$i]}$CONFIG &
    sleep 2s
done
#srun -p $1 --gres=gpu:1
