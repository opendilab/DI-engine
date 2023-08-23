from easydict import EasyDict

lunarlander_ppo_config = dict(
    exp_name='lunarlander_ppo_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        env_id='LunarLander-v2',
        n_evaluator_episode=5,
        stop_value=200,
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
            batch_size=64,
            learning_rate=3e-4,
            entropy_weight=0.01,
            adv_norm=True,
            value_norm=True,
        ),
        collect=dict(
            n_sample=512,
            discount_factor=0.99,
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
    policy=dict(type='ppo'),
)
lunarlander_ppo_create_config = EasyDict(lunarlander_ppo_create_config)
create_config = lunarlander_ppo_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy([main_config, create_config], seed=0)
