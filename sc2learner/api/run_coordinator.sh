#!/bin/bash
T=`date +%m%d%H%M`
cfg=config.yaml

ROOT=..
export PYTHONPATH=$ROOT:$ROOT/..:$PYTHONPATH

python -u $ROOT/api/coordinator_start.py \
  --config=${cfg} \
