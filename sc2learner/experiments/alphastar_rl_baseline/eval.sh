#!/usr/bin/env bash
work_path=$(dirname $0)
srun -p $1 --gres=gpu:1 python3 -u -m sc2learner.evaluate.evaluate \
    --config_path $work_path/eval.yaml \
