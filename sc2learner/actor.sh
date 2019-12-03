#!/usr/bin/env bash
for i in $(seq 0 $2); do
     srun -p $1  --gres=gpu:1 python3 -u -m sc2learner.bin.train_ppo --job_name=actor --learner_ip localhost &
done;
