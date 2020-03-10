#!/usr/bin/env bash
work_path=$(dirname $0)
python3 -u -m sc2learner.bin.train_ppo \
    --job_name learner \
    --config_path $work_path/config.yaml
