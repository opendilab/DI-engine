#!/bin/bash
set -e
T=`date +%m%d%H%M`
cfg=test_alphastar_actor_worker.yaml

ROOT=..
export PYTHONPATH=$ROOT:$ROOT/..:$PYTHONPATH


srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 \
    --job-name=actor \
python -u ./test_alphastar_actor_worker.py \
    --config_path=${cfg}