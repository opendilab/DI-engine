#!/bin/bash
set -e
T=`date +%m%d%H%M`
cfg=config.yaml

ROOT=..
export PYTHONPATH=$ROOT:$ROOT/..:$PYTHONPATH


srun --mpi=pmi2 -p $1 -w $2 \
     --job-name=actor \
python -m sc2learner.worker.actor.alphastar_actor_worker \
    --config_path=${cfg} $3
