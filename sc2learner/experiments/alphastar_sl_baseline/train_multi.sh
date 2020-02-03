#!/usr/bin/env bash
work_path=$(dirname $0)
ITER=00
srun --mpi=pmi2 -p $1 -n32 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 python3 -u -m sc2learner.bin.train_sl \
    --use_distributed True \
    --config_path $work_path/config.yaml \
    --replay_list $work_path/zerg_normal.txt \
#    --load_path $work_path/checkpoints/iterations_$ITER.pth.tar \
