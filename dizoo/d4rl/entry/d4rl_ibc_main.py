import os
from ding.entry import serial_pipeline_offline
from ding.config import read_config
from ding.utils import dist_init
from pathlib import Path
import torch
import torch.multiprocessing as mp

def offline_worker(rank, config, args):
    dist_init(rank=rank, world_size=torch.cuda.device_count())
    serial_pipeline_offline(config, seed=args.seed)

def train(args):
    # launch from anywhere
    config = Path(__file__).absolute().parent.parent / 'config' / args.config 
    config = read_config(str(config))
    config[0].exp_name = config[0].exp_name.replace('0', str(args.seed))
    if not config[0].policy.learn.multi_gpu:
        serial_pipeline_offline(config, seed=args.seed)
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29600"
        mp.spawn(offline_worker, nprocs=torch.cuda.device_count(), args=(config, args))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--config', '-c', type=str, default='hopper_medium_expert_ibc_config.py')
    args = parser.parse_args()
    train(args)
