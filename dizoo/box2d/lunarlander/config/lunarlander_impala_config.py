from easydict import EasyDict

lunarlander_impala_config = dict(
    exp_name='impala_log/lunarlander_impala_seed0',
    env=dict(
        env_id='LunarLander-v2',
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=3000,
    ),
    policy=dict(
        cuda=True,
        # (int) the trajectory length to calculate v-trace target
        unroll_len=32,
        random_collect_size=256,
        model=dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[64, 64],
        ),
        learn=dict(
            # (int) collect n_sample data, train model update_per_collect times
            # here we follow ppo serial pipeline
            update_per_collect=10,
            # (int) the number of data for a train iteration
            batch_size=128,
            grad_clip_type='clip_norm',
            clip_value=5,
            learning_rate=0.0003,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=0.5,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.0001,
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.99,
            # (float) additional discounting parameter
            lambda_=0.95,
            # (float) clip ratio of importance weights
            rho_clip_ratio=1.0,
            # (float) clip ratio of importance weights
            c_clip_ratio=1.0,
            # (float) clip ratio of importance sampling
            rho_pg_clip_ratio=1.0,
        ),
        collect=dict(
            # (int) collect n_sample data, train model update_per_collect times
            n_sample=32,
        ),
        eval=dict(evaluator=dict(eval_freq=500, )),
        other=dict(replay_buffer=dict(replay_buffer_size=1000, sliced=True), ),
    ),
)

lunarlander_impala_config = EasyDict(lunarlander_impala_config)
main_config = lunarlander_impala_config

lunarlander_impala_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='impala'),
    replay_buffer=dict(type='naive'),
)

lunarlander_impala_create_config = EasyDict(lunarlander_impala_create_config)
create_config = lunarlander_impala_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c lunarlander_impala_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
