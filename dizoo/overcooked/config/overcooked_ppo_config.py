from easydict import EasyDict
import torch.nn as nn

overcooked_ppo_config = dict(
    exp_name="overcooked_ppo_seed0",
    env=dict(
        collector_env_num=8,
        evaluator_env_num=10,
        n_evaluator_episode=10,
        concat_obs=False,  # stack 2 agents' obs in channel dim
        stop_value=80,
    ),
    policy=dict(
        cuda=True,
        multi_agent=True,
        action_space='discrete',
        model=dict(
            obs_shape=(26, 5, 4),
            action_shape=6,
            action_space='discrete',
        ),
        learn=dict(
            epoch_per_collect=4,
            batch_size=128,
            learning_rate=0.0005,
            entropy_weight=0.01,
            value_norm=True,
        ),
        collect=dict(
            n_sample=1024,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
    ),
)
overcooked_ppo_config = EasyDict(overcooked_ppo_config)
main_config = overcooked_ppo_config
cartpole_ppo_create_config = dict(
    env=dict(
        type='overcooked_game',
        import_names=['dizoo.overcooked.envs.overcooked_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
cartpole_ppo_create_config = EasyDict(cartpole_ppo_create_config)
create_config = cartpole_ppo_create_config


class OEncoder(nn.Module):

    def __init__(self, obs_shape):
        super(OEncoder, self).__init__()
        self.act = nn.ReLU()
        self.main = nn.Sequential(
            *[
                nn.Conv2d(obs_shape[0], 64, 3, 1, 1),
                self.act,
                nn.Conv2d(64, 64, 3, 1, 1),
                self.act,
                nn.Conv2d(64, 64, 3, 1, 1),
                self.act,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            ]
        )

    def forward(self, x):
        x = x.float()
        B, A = x.shape[:2]
        x = x.view(-1, *x.shape[2:])
        x = self.main(x)
        return x.view(B, A, 64)


if __name__ == "__main__":
    from ding.entry import serial_pipeline_onpolicy
    from ding.model.template import VAC
    m = main_config.policy.model
    encoder = OEncoder(obs_shape=m.obs_shape)
    model = VAC(obs_shape=m.obs_shape, action_shape=m.action_shape, action_space=m.action_space, encoder=encoder)
    serial_pipeline_onpolicy([main_config, create_config], seed=0, model=model)
