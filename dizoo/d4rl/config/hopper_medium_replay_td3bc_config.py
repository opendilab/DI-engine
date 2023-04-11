# You can conduct Experiments on D4RL with this config file through the following command:
# cd ../entry && python d4rl_td3_bc_main.py
from easydict import EasyDict

main_config = dict(
    exp_name='hopper_medium_replay_td3-bc_seed0',
    env=dict(
        env_id='hopper-medium-replay-v0',
        norm_obs=dict(
            use_norm=True, 
            offline_stats=dict(use_offline_stats=True, ),
        ),
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    policy=dict(
        model=dict(
            obs_shape=11,
            action_shape=3,
        ),
        learn=dict(
            train_epoch=30000,
            batch_size=256,
            learning_rate_actor=0.0003,
            learning_rate_critic=0.0003,
            actor_update_freq=2,
            noise=True,
            noise_sigma=0.2,
            noise_range={
                'min': -0.5,
                'max': 0.5
            },
            alpha=2.5,
        ),
        collect=dict(
            data_type='d4rl',
            data_path=None,
        ),
        eval=dict(evaluator=dict(eval_freq=10000, )),
        other=dict(replay_buffer=dict(replay_buffer_size=2000000, ), ),
    ),
)
main_config = EasyDict(main_config)
main_config = main_config
create_config = dict(
    env=dict(
        type='d4rl',
        import_names=['dizoo.d4rl.envs.d4rl_env'],
    ),
    env_manager=dict(
        cfg_type='BaseEnvManagerDict',
        type='base',
    ),
    policy=dict(
        type='td3_bc',
        import_names=['ding.policy.td3_bc'],
    ),
    replay_buffer=dict(type='naive', ),
)
create_config = EasyDict(create_config)
create_config = create_config
