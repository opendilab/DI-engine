#!/bin/bash
set -e
T=`date +%m%d%H%M`
work_path=$(dirname $0)
cfg=config.yaml

srun --mpi=pmi2 -p $1 -w $2 -n16 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=3 --job-name=learner \
    python3 -u -m sc2learner.train.train_rl \
        --config_path=$work_path/${cfg} \
        --job_name=learner \
