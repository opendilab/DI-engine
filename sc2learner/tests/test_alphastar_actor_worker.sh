#!/bin/bash
set -e
T=`date +%m%d%H%M`
cfg=test_alphastar_actor_worker.yaml

ROOT=..
export PYTHONPATH=$ROOT:$ROOT/..:$PYTHONPATH


srun --mpi=pmi2 -p $1 -w $2 \
     --job-name=actor \
python -u ./test_alphastar_actor_worker.py \
    --config_path=${cfg} $3
