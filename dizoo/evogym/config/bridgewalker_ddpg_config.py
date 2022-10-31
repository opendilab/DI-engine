from easydict import EasyDict

bridgewalker_ddpg_config = dict(
    exp_name='evogym_bridgewalker_ddpg_seed0',
    env=dict(
        env_id='BridgeWalker-v0',
        robot='speed_bot',
        robot_dir='../envs',
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=10,
        manager=dict(shared_memory=True, ),
        # The path to save the game replay
        # replay_path='./evogym_walker_ddpg_seed0/video',
    ),
    policy=dict(
        cuda=True,
        # load_path="./evogym_walker_ddpg_seed0/ckpt/ckpt_best.pth.tar",
        random_collect_size=1000,
        model=dict(
            obs_shape=59,
            action_shape=10,
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
bridgewalker_ddpg_config = EasyDict(bridgewalker_ddpg_config)
main_config = bridgewalker_ddpg_config

bridgewalker_ddpg_create_config = dict(
    env=dict(
        type='evogym',
        import_names=['dizoo.evogym.envs.evogym_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='ddpg',
        import_names=['ding.policy.ddpg'],
    ),
    replay_buffer=dict(type='naive', ),
)
bridgewalker_ddpg_create_config = EasyDict(bridgewalker_ddpg_create_config)
create_config = bridgewalker_ddpg_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c evogym_bridgewalker_ddpg_config.py -s 0 --env-step 1e7`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
