#!/usr/bin/env bash
srun -p $1  --gres=gpu:1 python3 -u -m sc2learner.bin.train_ppo --job_name learner
