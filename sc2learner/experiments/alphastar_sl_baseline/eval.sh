#!/usr/bin/env bash
work_path=$(dirname $0)
ITER=800
DIFFICULTY=1
srun -p $1 -w $2 --gres=gpu:1 python3 -u -m sc2learner.bin.evaluate \
    --config_path $work_path/eval.yaml \
    --load_path $work_path/checkpoints/iterations_$ITER.pth.tar \
    --difficulty $DIFFICULTY \
