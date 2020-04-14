#!/bin/bash
set -e
T=`date +%m%d%H%M`
cfg=config.yaml

ROOT=..
export PYTHONPATH=$ROOT:$ROOT/..:$PYTHONPATH

while true
do
    GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 \
        --job-name=actor \
    python -u $ROOT/api/fake_actor.py \
      --config=${cfg}
      
    echo finish one!
    sleep 5
done
