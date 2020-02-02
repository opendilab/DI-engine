#!/usr/bin/env bash
work_path=$(dirname $0)
ITER=00
srun -p $1 -w $2 --gres=gpu:1 python3 -u -m sc2learner.bin.train_sl \
    --use_distributed False \
    --config_path $work_path/config.yaml \
    --replay_list $work_path/zerg_normal.txt
#    --load_path $work_path/checkpoints/iterations_$ITER.pth.tar \
