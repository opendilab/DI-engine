from easydict import EasyDict

ant_ddpg_config = dict(
    exp_name='ant_ddpg_seed0',
    env=dict(
        env_id='Ant-v3',
        env_wrapper='mujoco_default',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=6000,
        manager=dict(shared_memory=False, ),
        # The path to save the game replay
        # replay_path='./ant_ddpg_seed0/video',
    ),
    policy=dict(
        cuda=True,
        load_path="./ant_ddpg_seed0/ckpt/ckpt_best.pth.tar",
        random_collect_size=25000,
        model=dict(
            obs_shape=111,
            action_shape=8,
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
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,  # discount_factor: 0.97-0.99
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
ant_ddpg_config = EasyDict(ant_ddpg_config)
main_config = ant_ddpg_config

ant_ddpg_create_config = dict(
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
ant_ddpg_create_config = EasyDict(ant_ddpg_create_config)
create_config = ant_ddpg_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c ant_ddpg_config.py -s 0 --env-step 1e7`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
