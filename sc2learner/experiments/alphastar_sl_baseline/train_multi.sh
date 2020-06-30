#!/usr/bin/env bash
work_path=$(dirname $0)
ITER=00
srun --mpi=pmi2 -p $1 -n2 --gres=gpu:2 --ntasks-per-node=2 --cpus-per-task=3 python3 -u -m sc2learner.train.train_sl \
    --use_distributed \
    --config_path $work_path/config.yaml \
    --noonly_evaluate \
    --replay_list /mnt/lustre/zhangmanyuan/as_data/602.zerg.128.zvz \
    --eval_replay_list /mnt/lustre/songguanglu/SenseStar/DATA/602.zerg.val \
#    --load_path $work_path/checkpoints/iterations_$ITER.pth.tar \
