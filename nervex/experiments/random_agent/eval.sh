#!/usr/bin/env bash
work_path=$(dirname $0)
DIFFICULTY=6
srun -p $1 python3 -u -m sc2learner.bin.evaluate \
    --config_path $work_path/eval.yaml \
    --replay_path _norand_randenv_"$DIFFICULTY" \
    --difficulty $DIFFICULTY \
