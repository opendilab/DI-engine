from easydict import EasyDict
from ding.entry import serial_pipeline

lunarlander_ppo_config = dict(
    exp_name='lunarlander_ppo_offpolicy',
    env=dict(
        manager=dict(shared_memory=True, reset_inplace=True),
        collector_env_num=8,
        evaluator_env_num=5,
        env_id='LunarLander-v2',
        n_evaluator_episode=5,
        stop_value=200,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=8,
            action_shape=4,
        ),
        learn=dict(
            update_per_collect=4,
            batch_size=64,
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
lunarlander_ppo_config = EasyDict(lunarlander_ppo_config)
main_config = lunarlander_ppo_config
lunarlander_ppo_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo_offpolicy'),
)
lunarlander_ppo_create_config = EasyDict(lunarlander_ppo_create_config)
create_config = lunarlander_ppo_create_config

if __name__ == "__main__":
    serial_pipeline([main_config, create_config], seed=0)
