from easydict import EasyDict

hopper_d4pg_default_config = dict(
    exp_name='hopper_d4pg_seed0',
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=4,
        evaluator_env_num=4,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=5000,
    ),
    policy=dict(
        cuda=True,
        priority=True,
        nstep=5,
        random_collect_size=10000,
        model=dict(
            obs_shape=11,
            action_shape=3,
            actor_head_hidden_size=512,
            critic_head_hidden_size=512,
            action_space='regression',
            critic_head_type='categorical',
            v_min=0,
            v_max=1000,  # [1000, 3000]
            n_atom=51,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_actor=3e-4,
            learning_rate_critic=3e-4,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            actor_update_freq=1,
            noise=False,
        ),
        collect=dict(
            n_sample=8,
            unroll_len=1,
            noise_sigma=0.2,  # [0.1, 0.2]
        ),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    )
)
hopper_d4pg_default_config = EasyDict(hopper_d4pg_default_config)
main_config = hopper_d4pg_default_config

hopper_d4pg_default_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='d4pg',
        import_names=['ding.policy.d4pg'],
    ),
)
hopper_d4pg_default_create_config = EasyDict(hopper_d4pg_default_create_config)
create_config = hopper_d4pg_default_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)
