work_path=$(dirname $0)
python3 -u -m sc2learner.bin.train_ppo \
    --job_name actor_manager \
    --config_path $work_path/config.yaml \
