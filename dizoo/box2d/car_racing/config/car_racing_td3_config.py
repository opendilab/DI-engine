from easydict import EasyDict

car_racing_td3_config = dict(
    exp_name='car_racing_td3',
    env=dict(
        env_id='CarRacing-v0',
        collector_env_num=8,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range.
        is_train=True,
        frame_stack=4,
        act_scale=True,
        rew_clip=True,
        replay_path='./replay',
        n_evaluator_episode=5,
        stop_value=-250,
    ),
    policy=dict(
        cuda=False,
        priority=False,
        random_collect_size=800,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=3,
            encoder_hidden_size_list=[64, 64, 128],
            critic_head_hidden_size=128,
            actor_head_hidden_size=128,
            action_space='regression',
        ),
        learn=dict(
            update_per_collect=2,
            batch_size=128,
            learning_rate_actor=0.001,
            learning_rate_critic=0.001,
            ignore_done=True,
            actor_update_freq=2,
            noise=True,
            noise_sigma=0.1,
            noise_range=dict(
                min=-0.5,
                max=0.5,
            ),
        ),
        collect=dict(
            n_sample=48,
            noise_sigma=0.1,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=100, ), ),
        other=dict(replay_buffer=dict(replay_buffer_size=20000, ), ),
    ),
)
car_racing_td3_config = EasyDict(car_racing_td3_config)
main_config = car_racing_td3_config

car_racing_td3_create_config = dict(
    env=dict(
        type='car_racing',
        import_names=['dizoo.box2d.car_racing.envs.car_racing_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='td3'),
)
car_racing_td3_create_config = EasyDict(car_racing_td3_create_config)
create_config = car_racing_td3_create_config
