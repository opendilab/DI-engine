#!/usr/bin/env bash
work_path=$(dirname $0)
ITER=00
srun -p $1 -w $2 --gres=gpu:1 python3 -u -m sc2learner.bin.train_sl \
    --use_distributed false \
    --config_path $work_path/config.yaml \
    --only_evaluate false \
    --replay_list $work_path/zerg_500.txt \
    --eval_replay_list $work_path/zerg_8.txt \
#    --load_path $work_path/checkpoints/iterations_$ITER.pth.tar \
