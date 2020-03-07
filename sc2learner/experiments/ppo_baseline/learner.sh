#!/usr/bin/env bash
work_path=$(dirname $0)
if [ "$3" == "" ]
then
    checkpoint_arg=""
else
    checkpoint_arg="--load_path $work_path/checkpoints/iterations_$3.pth.tar"
fi
srun -p $1 -w $2 --job-name=learner --gres=gpu:1 python3 -u -m sc2learner.bin.train_ppo \
    --job_name learner \
    --config_path $work_path/config.yaml \
    $checkpoint_arg \
#    --data_load_path $work_path/pre_data
