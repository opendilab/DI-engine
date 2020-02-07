#!/usr/bin/env bash
work_path=$(dirname $0)
ITER=00
srun -p $1 -w $2 --job-name=learner --gres=gpu:1 python3 -u -m sc2learner.bin.train_ppo \
    --job_name learner \
    --config_path $work_path/config.yaml \
#    --load_path $work_path/checkpoints/iterations_$ITER.pth.tar \
#    --data_load_path $work_path/pre_data
