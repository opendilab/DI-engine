from typing import Union, Optional, List, Any, Tuple
import collections
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import treetensor.torch as ttorch

import gym
from gym import spaces

from ditk import logging
from functools import partial
from tensorboardX import SummaryWriter
from copy import deepcopy

from ding.envs import get_vec_env_setting, create_env_manager, DingEnvWrapper, EvalEpisodeReturnEnv
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.config import read_config, compile_config
from ding.policy import create_policy, PolicyFactory
from ding.reward_model import create_reward_model
from ding.utils import set_pkg_seed
from ding.model import VAC

from easydict import EasyDict

my_env_ppo_config = dict(
    exp_name='my_env_ppo_seed0',
    env=dict(
        collector_env_num=4,
        evaluator_env_num=4,
        n_evaluator_episode=4,
        stop_value=195,
    ),
    policy=dict(
        cuda=True,
        action_space='discrete',
        model=dict(
            obs_shape=None,
            action_shape=2,
            action_space='discrete',
            critic_head_hidden_size=3314,
            actor_head_hidden_size=3314,
        ),
        learn=dict(
            epoch_per_collect=2,
            batch_size=64,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            learner=dict(hook=dict(save_ckpt_after_iter=100)),
        ),
        collect=dict(
            n_sample=256, unroll_len=1, discount_factor=0.9, gae_lambda=0.95, collector=dict(transform_obs=True, )
        ),
        eval=dict(evaluator=dict(eval_freq=100, ), ),
    ),
)
my_env_ppo_config = EasyDict(my_env_ppo_config)
main_config = my_env_ppo_config
my_env_ppo_create_config = dict(
    # env=dict(
    #     type='my_env',
    #     import_names=['dizoo.classic_control.my_env.envs.my_env_env'],
    # ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
)
my_env_ppo_create_config = EasyDict(my_env_ppo_create_config)
create_config = my_env_ppo_create_config


class MyEnv(gym.Env):

    def __init__(self, seq_len=5, feature_dim=10, chart_seq_len=10):
        super().__init__()

        # Define the action space
        self.action_space = spaces.Discrete(2)

        # Define the observation space
        self.observation_space = spaces.Tuple(
            (
                spaces.Dict(
                    {
                        'k1': spaces.Box(low=0, high=np.inf, shape=(1, ), dtype=np.float32),
                        'k2': spaces.Box(low=-1, high=1, shape=(1, ), dtype=np.float32),
                    }
                ), spaces.Box(low=-np.inf, high=np.inf, shape=(seq_len, feature_dim), dtype=np.float32),
                spaces.Box(low=0, high=255, shape=(chart_seq_len, chart_seq_len, 3), dtype=np.uint8),
                spaces.Box(low=0, high=np.array([np.inf, 3]), shape=(2, ), dtype=np.float32)
            )
        )

    def reset(self):
        # Generate a random initial state
        return self.observation_space.sample()

    def step(self, action):
        # Compute the reward and done flag (which are not used in this example)
        reward = np.random.uniform(low=0.0, high=1.0)

        done = False
        if np.random.uniform(low=0.0, high=1.0) > 0.7:
            done = True

        info = {}

        # Return the next state, reward, and done flag
        return self.observation_space.sample(), reward, done, info


def ding_env_maker():
    return DingEnvWrapper(
        MyEnv(), cfg={'env_wrapper': [
            lambda env: EvalEpisodeReturnEnv(env),
        ]}, seed_api=False
    )


class Encoder(nn.Module):

    def __init__(self, feature_dim, seq_len, chart_seq_len):
        super(Encoder, self).__init__()

        # Define the networks for each input type
        self.fc_net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU())
        self.transformer_net = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=2)
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc_net_2 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU())

    def forward(self, inputs):
        # Unpack the input tuple
        dict_input, transformer_input, conv_input, fc_input = inputs

        B = fc_input.shape[0]

        # Pass each input through its corresponding network
        dict_output = self.fc_net(torch.stack([dict_input['k1'], dict_input['k2']], dim=1))
        transformer_output = self.transformer_net(transformer_input)
        conv_output = self.conv_net(conv_input.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        fc_output = self.fc_net_2(fc_input)

        # Concatenate the outputs along the feature dimension
        encoded_output = torch.cat(
            [dict_output.view(B, -1),
             transformer_output.view(B, -1),
             conv_output.view(B, -1),
             fc_output.view(B, -1)],
            dim=1
        )

        return encoded_output


def pipeline(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry on-policy RL.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
        - max_env_step (:obj:`Optional[int]`): Maximum collected environment interaction steps.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)
    # Create main components: env, policy

    collector_env = create_env_manager(cfg.env.manager, [ding_env_maker for _ in range(cfg.env.collector_env_num)])
    evaluator_env = create_env_manager(cfg.env.manager, [ding_env_maker for _ in range(cfg.env.evaluator_env_num)])

    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    encoder = Encoder(seq_len=5, feature_dim=10, chart_seq_len=10)

    model = VAC(encoder=encoder, **cfg.policy.model)

    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = create_serial_collector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, None, policy.command_mode
    )

    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')

    while True:
        collect_kwargs = commander.step()
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, eval_info = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        # Collect data by default config n_sample/n_episode
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)

        # Learn policy from collected data
        learner.train(new_data, collector.envstep)
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
    import time
    import pickle
    import numpy as np
    with open(os.path.join(cfg.exp_name, 'result.pkl'), 'wb') as f:
        eval_value_raw = [d['eval_episode_return'] for d in eval_info]
        final_data = {
            'stop': stop,
            'env_step': collector.envstep,
            'train_iter': learner.train_iter,
            'eval_value': np.mean(eval_value_raw),
            'eval_value_raw': eval_value_raw,
            'finish_time': time.ctime(),
        }
        pickle.dump(final_data, f)
    return policy


if __name__ == "__main__":
    pipeline((main_config, create_config), seed=0)
