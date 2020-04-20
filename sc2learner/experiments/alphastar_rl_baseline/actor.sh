#!/bin/bash
set -e
T=`date +%m%d%H%M`
work_path=$(dirname $0)
cfg=config.yaml

srun --mpi=pmi2 -p $1 $2 --job-name=actor \
    python3 -u -m sc2learner.train.train_rl \
        --config_path=${cfg} \
        --job_name=actor \
