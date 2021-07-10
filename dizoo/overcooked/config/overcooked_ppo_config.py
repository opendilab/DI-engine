from easydict import EasyDict
from numpy.core.shape_base import stack
from ding.entry import serial_pipeline

overcooked_ppo_config = dict(
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=10,
		stack_obs=True,
		action_mask=False,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=(10, 4, 26),
            action_shape=(2,6),
        ),
        learn=dict(
            update_per_collect=4,
            batch_size=16,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            nstep=1,
            nstep_return=False,
            adv_norm=True,
        ),
        collect=dict(
            n_sample=128,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
    ),
)
overcooked_ppo_config = EasyDict(overcooked_ppo_config)
main_config = overcooked_ppo_config
overcooked_ppo_create_config = dict(
    env=dict(
        type='overcooked',
        import_names=['dizoo.overcooked.envs.overcooked_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
)
overcooked_ppo_create_config = EasyDict(overcooked_ppo_create_config)
create_config = overcooked_ppo_create_config

if __name__ == "__main__":
    serial_pipeline([main_config, create_config], seed=0)
