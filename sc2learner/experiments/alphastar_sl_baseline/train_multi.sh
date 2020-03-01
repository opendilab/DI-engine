#!/usr/bin/env bash
work_path=$(dirname $0)
ITER=00
srun --mpi=pmi2 -p $1 -n8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=3 python3 -u -m sc2learner.bin.train_sl \
    --use_distributed \
    --config_path $work_path/config.yaml \
    --noonly_evaluate \
    --replay_list $work_path/zerg_500.txt \
    --eval_replay_list $work_path/zerg_8.txt \
#    --load_path $work_path/checkpoints/iterations_$ITER.pth.tar \
