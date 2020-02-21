#!/usr/bin/env bash
work_path=$(dirname $0)
ITER=0
for i in $(seq 1 $3); do
     srun -p $1 -w $2 --job-name=actor python3 -u -m sc2learner.bin.train_ppo \
        --job_name actor \
        --seed "$(($4 + $i))" \
        --node_name $2 \
        --config_path $work_path/config.yaml &\
        #--load_path $work_path/checkpoints/iterations_$ITER.pth.tar &
done;
