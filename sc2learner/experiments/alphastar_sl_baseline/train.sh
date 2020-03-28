#!/usr/bin/env bash
work_path=$(dirname $0)
ITER=00
srun -p $1 --gres=gpu:1 python3 -u -m sc2learner.train.train_sl \
    --nouse_distributed \
    --config_path $work_path/config.yaml \
    --noonly_evaluate \
    --replay_list /mnt/lustre/share_data/niuyazhe/Zerg_None_None_3500_train_5200.txt \
    --eval_replay_list /mnt/lustre/share_data/niuyazhe/Zerg_None_None_3500_val_274.txt \
#    --load_path $work_path/checkpoints/iterations_$ITER.pth.tar \
