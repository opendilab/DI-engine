#!/usr/bin/env bash
work_path=$(dirname $0)
ITER1=570000
ITER2=300000
srun -p $1 --gres=gpu:1 python3 -u -m sc2learner.bin.eval_selfplay \
    --config_path $work_path/eval_selfplay.yaml \
    --agent1_load_path $work_path/checkpoints/iterations_$ITER1.pth.tar \
    --agent2_load_path $work_path/checkpoints/iterations_$ITER2.pth.tar \
    --replay_path "$ITER1"_"$ITER2" \
