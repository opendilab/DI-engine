#!/usr/bin/env bash
work_path=$(dirname $0)
for i in $(seq 1 $2); do
     srun -p $1 python3 -u -m sc2learner.bin.train_ppo \
        --job_name actor \
        --config_path $work_path/config.yaml \
        &
done;
