#!/bin/bash
T=`date +%m%d%H%M`
cfg=config.yaml
agent_id=0

ROOT=..
export PYTHONPATH=$ROOT:$ROOT/..:$PYTHONPATH

srun --mpi=pmi2 -p $1 -w $2 \
    --job-name=learner \
python -u $ROOT/api/learner_api.py \
  --config=${cfg}
