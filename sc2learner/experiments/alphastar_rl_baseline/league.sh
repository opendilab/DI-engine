#!/bin/bash
set -e
T=`date +%m%d%H%M`
work_path=$(dirname $0)
cfg=config.yaml

srun -p $1 --job-name=league \
    python3 -u -m sc2learner.system.league_manager_start \
        --config $work_path/${cfg} \

