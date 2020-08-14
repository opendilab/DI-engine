#!/bin/bash
T=`date +%m%d%H%M`
work_path=$(dirname $0)
cfg=config.yaml

python3 -u -m sc2learner.system.coordinator_start \
    --config $work_path/$cfg \
