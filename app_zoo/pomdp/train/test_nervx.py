from copy import deepcopy
import numpy as np
import torch
from pprint import pprint
from nervex.model import FCDiscreteNet
from nervex.model.actor_critic import FCValueAC
from nervex.entry import serial_pipeline
from app_zoo.classic_control.cartpole.entry import cartpole_dqn_default_config
from app_zoo.pomdp.entry import pomdp_dqn_default_config, pomdp_ppo_default_config
from app_zoo.atari.entry.atari_serial_baseline import pong_dqn_default_config
from app_zoo.pomdp.envs.atari_env import PomdpAtariEnv


def train_dqn(args):
    config = deepcopy(pomdp_ppo_default_config)
    # config = deepcopy(pomdp_dqn_default_config)
    config.env.env_id = args.env
    config.policy.embedding_dim = args.embedding_dim
    if args.test:
        config["env"]["is_train"] = False
        eval_dqn(config, args)
        return 0
    serial_pipeline(config, seed=args.seed)


def eval_dqn(config, args):
    # config.env.render = True
    env = PomdpAtariEnv(config.env)
    # model = FCDiscreteNet((512,), (6,), embedding_dim=128)      # dqn
    model = FCValueAC(obs_dim=(512,), action_dim=6, embedding_dim=args.embedding_dim)  # AC

    model.load_state_dict(torch.load(args.ckpt, map_location=torch.device('cpu'))['model'])

    for iter in range(args.test_iter):
        done = False
        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        cum_reward = 0
        while not done:
            logit = model.forward(obs, mode="compute_action")["logit"]  # AC
            # logit = model.forward(obs)["logit"]    # dqn
            action = logit.argmax()
            timestep = env.step(action.numpy())
            obs, reward, done, info = timestep
            obs = torch.tensor(obs, dtype=torch.float32)
            cum_reward += reward[0]

        print("Mean reward: ", cum_reward)


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

