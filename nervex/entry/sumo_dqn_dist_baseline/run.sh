work_path=$(dirname $0)
RES_PREFIX=("" "")
CMD=("python3 -u -m nervex.system.coordinator_start" "python3 -u -m nervex.system.manager_start" \
     "python3 -u -m nervex.system.league_manager_start" "python3 -u -m nervex.system.learner_start" \
     "python3 -u -m nervex.system.actor_start")
CONFIG=" --config $work_path/sumo_dqn_dist_default_config.yaml"

for ((i=0;i<${#CMD[@]};i++))
do
    ${CMD[$i]}$CONFIG &
    sleep 2s
done
#srun -p $1 --gres=gpu:1
