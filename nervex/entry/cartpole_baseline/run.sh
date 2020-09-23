work_path=$(dirname $0)
srun -p $1 --gres=gpu:1 python3 -u ../cartpole_dqn_main.py\
    --config_path $work_path/cartpole_dqn_default_config.yaml
