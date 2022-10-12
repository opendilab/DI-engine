from easydict import EasyDict

halfcheetah_ddpg_config = dict(
    env=dict(
        exp_name='halfcheetah_ddpg_seed0',
        env_id='HalfCheetah-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=11000,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=25000,
        model=dict(
            obs_shape=17,
            action_shape=6,
            twin_critic=False,
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
            action_space='regression',
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_actor=1e-3,
            learning_rate_critic=1e-3,
            ignore_done=True,
            target_theta=0.005,
            discount_factor=0.99,
            actor_update_freq=1,
            noise=False,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
            noise_sigma=0.1,
        ),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    )
)
halfcheetah_ddpg_config = EasyDict(halfcheetah_ddpg_config)
main_config = halfcheetah_ddpg_config

halfcheetah_ddpg_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='ddpg',
        import_names=['ding.policy.ddpg'],
    ),
    replay_buffer=dict(type='naive', ),
)
halfcheetah_ddpg_create_config = EasyDict(halfcheetah_ddpg_create_config)
create_config = halfcheetah_ddpg_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c halfcheetah_ddpg_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)