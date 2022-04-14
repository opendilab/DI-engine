from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 8
lunarlander_ppo_rnd_config = dict(
    exp_name='lunarlander_rnd_onppo_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        env_id='LunarLander-v2',
        n_evaluator_episode=8,
        stop_value=200,
    ),
    reward_model=dict(
        intrinsic_reward_type='add',  # 'assign'
        learning_rate=5e-4,
        obs_shape=8,
        batch_size=320,
        update_per_collect=4,
    ),
    policy=dict(
        recompute_adv=True,
        cuda=True,
        action_space='discrete',
        model=dict(
            obs_shape=8,
            action_shape=4,
            action_space='discrete',
        ),
        learn=dict(
            epoch_per_collect=10,
            update_per_collect=1,
            batch_size=64,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
        ),
        collect=dict(
            # n_sample=128,
            collector_env_num=collector_env_num,
            n_sample=int(64 * collector_env_num),
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
    ),
)
lunarlander_ppo_rnd_config = EasyDict(lunarlander_ppo_rnd_config)
main_config = lunarlander_ppo_rnd_config
lunarlander_ppo_rnd_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
    reward_model=dict(type='rnd')
)
lunarlander_ppo_rnd_create_config = EasyDict(lunarlander_ppo_rnd_create_config)
create_config = lunarlander_ppo_rnd_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_reward_model
    serial_pipeline_reward_model([main_config, create_config], seed=0)
