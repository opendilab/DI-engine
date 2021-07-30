from easydict import EasyDict

cartpole_acer_config = dict(
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[64, 64],
        ),
        # (int) the trajectory length to calculate Q retrace target
        unroll_len=32,
        learn=dict(
            # (int) collect n_sample data, train model update_per_collect times
            # here we follow ppo serial pipeline
            update_per_collect=4,
            # (int) the number of data for a train iteration
            batch_size=16,
            learning_rate_actor=0.0005,
            learning_rate_critic=0.0005,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            # entropy_weight=0.0001,
            entropy_weight=0.0,
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.9,
            # (float) additional discounting parameter
            # (int) the trajectory length to calculate v-trace target
            # (float) clip ratio of importance weights
            trust_region=True,
            c_clip_ratio=10,
            # (float) clip ratio of importance sampling
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration times
            n_sample=16,
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.9,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=200, )),
        other=dict(replay_buffer=dict(replay_buffer_size=10000, ), ),
    ),
)

cartpole_acer_config = EasyDict(cartpole_acer_config)
main_config = cartpole_acer_config

cartpole_acer_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='acer'),
)

cartpole_acer_create_config = EasyDict(cartpole_acer_create_config)
create_config = cartpole_acer_create_config
