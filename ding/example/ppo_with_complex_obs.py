from typing import Dict
import os
import torch
import torch.nn as nn
import numpy as np
import gym
from gym import spaces
from ditk import logging
from ding.envs import DingEnvWrapper, EvalEpisodeReturnWrapper, \
    BaseEnvManagerV2
from ding.config import compile_config
from ding.policy import PPOPolicy
from ding.utils import set_pkg_seed
from ding.model import VAC
from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import multistep_trainer, StepCollector, interaction_evaluator, CkptSaver, \
    gae_estimator, online_logger
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
            critic_head_hidden_size=138,
            actor_head_hidden_size=138,
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
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
)
my_env_ppo_create_config = EasyDict(my_env_ppo_create_config)
create_config = my_env_ppo_create_config


class MyEnv(gym.Env):

    def __init__(self, seq_len=5, feature_dim=10, image_size=(10, 10, 3)):
        super().__init__()

        # Define the action space
        self.action_space = spaces.Discrete(2)

        # Define the observation space
        self.observation_space = spaces.Dict(
            (
                {
                    'key_0': spaces.Dict(
                        {
                            'k1': spaces.Box(low=0, high=np.inf, shape=(1, ), dtype=np.float32),
                            'k2': spaces.Box(low=-1, high=1, shape=(1, ), dtype=np.float32),
                        }
                    ),
                    'key_1': spaces.Box(low=-np.inf, high=np.inf, shape=(seq_len, feature_dim), dtype=np.float32),
                    'key_2': spaces.Box(low=0, high=255, shape=image_size, dtype=np.uint8),
                    'key_3': spaces.Box(low=0, high=np.array([np.inf, 3]), shape=(2, ), dtype=np.float32)
                }
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
            lambda env: EvalEpisodeReturnWrapper(env),
        ]}
    )


class Encoder(nn.Module):

    def __init__(self, feature_dim: int):
        super(Encoder, self).__init__()

        # Define the networks for each input type
        self.fc_net_1_k1 = nn.Sequential(nn.Linear(1, 8), nn.ReLU())
        self.fc_net_1_k2 = nn.Sequential(nn.Linear(1, 8), nn.ReLU())
        self.fc_net_1 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        """
        Implementation of transformer_encoder refers to Vision Transformer (ViT) code:
            https://arxiv.org/abs/2010.11929
            https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html
        """
        self.class_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_fc_net = nn.Sequential(nn.Flatten(), nn.Linear(3200, 64), nn.ReLU())

        self.fc_net_2 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Flatten())

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Unpack the input tuple
        dict_input = inputs['key_0']  # dict{key:(B)}
        transformer_input = inputs['key_1']  # (B, seq_len, feature_dim)
        conv_input = inputs['key_2']  # (B, H, W, 3)
        fc_input = inputs['key_3']  # (B, X)

        B = fc_input.shape[0]

        # Pass each input through its corresponding network
        dict_output = self.fc_net_1(
            torch.cat(
                [self.fc_net_1_k1(dict_input['k1'].unsqueeze(-1)),
                 self.fc_net_1_k2(dict_input['k2'].unsqueeze(-1))],
                dim=1
            )
        )

        batch_class_token = self.class_token.expand(B, -1, -1)
        transformer_output = self.transformer_encoder(torch.cat([batch_class_token, transformer_input], dim=1))
        transformer_output = transformer_output[:, 0]

        conv_output = self.conv_fc_net(self.conv_net(conv_input.permute(0, 3, 1, 2)))
        fc_output = self.fc_net_2(fc_input)

        # Concatenate the outputs along the feature dimension
        encoded_output = torch.cat([dict_output, transformer_output, conv_output, fc_output], dim=1)

        return encoded_output


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    ding_init(cfg)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = BaseEnvManagerV2(
            env_fn=[ding_env_maker for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
        )
        evaluator_env = BaseEnvManagerV2(
            env_fn=[ding_env_maker for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        encoder = Encoder(feature_dim=10)
        model = VAC(encoder=encoder, **cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(gae_estimator(cfg, policy.collect_mode))
        task.use(multistep_trainer(policy.learn_mode, log_freq=50))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=100))
        task.use(online_logger(train_show_freq=3))
        task.run()


if __name__ == "__main__":
    main()
