from easydict import EasyDict

gym_hybrid_ddpg_config = dict(
    exp_name='gym_hybrid_ddpg_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range [-1, 1].
        act_scale=True,
        env_id='Moving-v0',  # ['Sliding-v0', 'Moving-v0']
        n_evaluator_episode=5,
        stop_value=1.8,
        save_replay_gif=False,
        replay_path_gif=None,
    ),
    policy=dict(
        cuda=True,
        priority=False,
        random_collect_size=0,  # hybrid action space not support random collect now
        action_space='hybrid',
        model=dict(
            obs_shape=10,
            action_shape=dict(
                action_type_shape=3,
                action_args_shape=2,
            ),
            twin_critic=False,
            action_space='hybrid',
        ),
        learn=dict(
            update_per_collect=10,  # 5~10
            batch_size=32,
            discount_factor=0.99,
            learning_rate_actor=0.0003,  # 0.001 ~ 0.0003
            learning_rate_critic=0.001,
            actor_update_freq=1,
            noise=False,
        ),
        collect=dict(
            n_sample=32,
            noise_sigma=0.1,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=1000, ), ),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.1,
                decay=100000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, ),
        ),
    ),
)
gym_hybrid_ddpg_config = EasyDict(gym_hybrid_ddpg_config)
main_config = gym_hybrid_ddpg_config

gym_hybrid_ddpg_create_config = dict(
    env=dict(
        type='gym_hybrid',
        import_names=['dizoo.gym_hybrid.envs.gym_hybrid_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ddpg'),
)
gym_hybrid_ddpg_create_config = EasyDict(gym_hybrid_ddpg_create_config)
create_config = gym_hybrid_ddpg_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c gym_hybrid_ddpg_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)
