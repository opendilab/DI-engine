from easydict import EasyDict
import torch
import torch.nn as nn
from ding.model.common import FCEncoder, ReparameterizationHead

bipedalwalker_ppo_config = dict(
    exp_name='bipedalwalker_ppopg',
    env=dict(
        env_id='BipedalWalker-v3',
        collector_env_num=8,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=500,
        rew_clip=True,
    ),
    policy=dict(
        cuda=True,
        action_space='continuous',
        model=dict(
            obs_shape=24,
            action_shape=4,
        ),
        learn=dict(
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
            entropy_weight=0.0001,
            clip_ratio=0.2,
            adv_norm=True,
        ),
        collect=dict(
            n_episode=16,
            discount_factor=0.99,
            collector=dict(get_train_sample=True),
        ),
    ),
)
bipedalwalker_ppo_config = EasyDict(bipedalwalker_ppo_config)
main_config = bipedalwalker_ppo_config
bipedalwalker_ppo_create_config = dict(
    env=dict(
        type='bipedalwalker',
        import_names=['dizoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo_pg'),
    collector=dict(type='episode'),
)
bipedalwalker_ppo_create_config = EasyDict(bipedalwalker_ppo_create_config)
create_config = bipedalwalker_ppo_create_config


class PPOPGContinuousModel(nn.Module):

    def __init__(self, obs_shape, action_shape):
        super(PPOPGContinuousModel, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_shape, 64), nn.Tanh())
        self.head = ReparameterizationHead(
            hidden_size=64,
            output_size=action_shape,
            layer_num=2,
            sigma_type='conditioned',
            activation=nn.Tanh(),
        )

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.head(x)
        return {'logit': x}


if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c bipedalwalker_ppo_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    from copy import deepcopy
    for seed in [1, 2, 3]:
        new_main_config = deepcopy(main_config)
        new_main_config.exp_name += "_seed{}".format(seed)
        model = PPOPGContinuousModel(new_main_config.policy.model.obs_shape, new_main_config.policy.model.action_shape)
        serial_pipeline_onpolicy(
            [new_main_config, deepcopy(create_config)], seed=seed, max_env_step=int(5e6), model=model
        )
