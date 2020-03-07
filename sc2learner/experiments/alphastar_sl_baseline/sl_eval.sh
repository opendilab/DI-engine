#!/usr/bin/env bash
work_path=$(dirname $0)
ITER=1400
srun --mpi=pmi2 -p $1 -n2 --gres=gpu:2 --ntasks-per-node=2 --cpus-per-task=3 python3 -u -m sc2learner.bin.train_sl \
    --use_distributed \
    --config_path $work_path/config.yaml \
    --only_evaluate \
    --replay_list $work_path/zerg_500.txt \
    --eval_replay_list $work_path/zerg_1.txt \
    --load_path $work_path/checkpoints/iterations_$ITER.pth.tar \
