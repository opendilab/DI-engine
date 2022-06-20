from easydict import EasyDict

cartpole_fqf_config = dict(
    exp_name='cartpole_fqf_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
        replay_path='cartpole_fqf_seed0/video',
    ),
    policy=dict(
        cuda=False,
        priority=True,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            num_quantiles=32,
            quantile_embedding_size=64,
        ),
        discount_factor=0.97,
        nstep=1,
        learn=dict(
            update_per_collect=3,
            batch_size=64,
            learning_rate_fraction=0.0001,
            learning_rate_quantile=0.0001,
            target_update_freq=100,
            ent_coef=0,
        ),
        collect=dict(
            n_sample=80,
            unroll_len=1,
        ),
        eval=dict(evaluator=dict(eval_freq=40, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ), replay_buffer=dict(replay_buffer_size=20000, )
        ),
    ),
)
cartpole_fqf_config = EasyDict(cartpole_fqf_config)
main_config = cartpole_fqf_config
cartpole_fqf_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='fqf'),
)
cartpole_fqf_create_config = EasyDict(cartpole_fqf_create_config)
create_config = cartpole_fqf_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c cartpole_fqf_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)