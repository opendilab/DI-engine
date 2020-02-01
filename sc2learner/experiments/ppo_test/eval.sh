#!/usr/bin/env bash
work_path=$(dirname $0)
ITER=570000
DIFFICULTY=6
srun -p $1 --gres=gpu:1 python3 -u -m sc2learner.bin.evaluate \
    --config_path $work_path/eval.yaml \
    --load_path $work_path/checkpoints/iterations_$ITER.pth.tar \
    --replay_path _norand_randenv_iteration_"$ITER"_"$DIFFICULTY" \
    --difficulty $DIFFICULTY
