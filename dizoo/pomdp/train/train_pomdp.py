from ast import iter_child_nodes
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from copy import deepcopy
import spinup.algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from dizoo.classic_control.cartpole.entry import cartpole_dqn_default_config
from dizoo.pomdp.entry import pomdp_dqn_default_config, pomdp_ppo_default_config
from dizoo.atari.entry.atari_serial_baseline import pong_dqn_default_config
from dizoo.pomdp.envs.atari_env import PomdpAtariEnv, PomdpEnv
from spinup import ppo_pytorch
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train(args):
    config = deepcopy(pomdp_ppo_default_config)
    config.env.env_id = args.env

    if args.test:
        test(config, args)
        return 0

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir="./data")

    train_env = PomdpEnv(config.env)
    ppo_pytorch(
        lambda: train_env,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs
    )


def test(config, args):

    config.env.is_train = False
    config.env.render = False
    train_env = PomdpEnv(config.env)
    ac = torch.load(args.ckpt)

    for iter in range(1, 6):
        train_env.seed(iter)
        obs = train_env.reset()
        done = False
        cum_reward = 0
        while not done:
            action = ac.act(torch.as_tensor(obs, dtype=torch.float32))
            obs, reward, done, info = train_env.step(action)
            cum_reward += reward
        print(cum_reward)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pong-ramNoFrameskip-v4')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--steps', type=int, default=16_000)
    parser.add_argument('--epochs', type=int, default=100_000_000)
    parser.add_argument('--exp_name', type=str, default='ppo2')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    train(args)
