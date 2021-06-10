import sys
# try:
#     sys.path.remove('d:\\学习\\computer\\科研\\rl\\nervex_dev_config')
# except:
#     a=1
from copy import deepcopy
from nervex.entry import serial_pipeline
from app_zoo.smac.model_zoo.config import smac_5m6m_collaq_main_config, smac_5m6m_collaq_create_config


def train_dqn(args):
    config = [smac_5m6m_collaq_main_config, smac_5m6m_collaq_create_config]
    serial_pipeline(config, seed=args.seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pong-ramNoFrameskip-v4')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--test_iter', type=int, default=5)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ckpt', type=str)
    args = parser.parse_args()

    train_dqn(args)
