work_path=$(dirname $0)
srun -p $1 --gres=gpu:1 python3 -u ../sumo_dqn_main.py\
    --config_path $work_path/sumo_dqn_default_config.yaml 
