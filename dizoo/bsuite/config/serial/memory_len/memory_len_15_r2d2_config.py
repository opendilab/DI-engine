from easydict import EasyDict

memory_len_r2d2_config = dict(
    exp_name='memory_len_15_r2d2_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=1,
        n_evaluator_episode=20,
        env_id='memory_len/15',  # this environment configuration is 30 'memory steps' long
        stop_value=1.,
        manager=dict(shared_memory=False, )
    ),
    policy=dict(
        cuda=True,
        priority=True,
        priority_IS_weight=True,
        model=dict(
            obs_shape=3,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        discount_factor=0.997,  # discount_factor: 0.97-0.99
        burnin_step=1,  # fix to 1 since early steps are the most important
        nstep=3,
        unroll_len=40,  # for better converge should be unroll_len > 'memory steps' = 30
        learn=dict(
            update_per_collect=8,
            batch_size=64,
            learning_rate=0.0005,
            target_update_theta=0.001,
        ),
        collect=dict(
            each_iter_n_sample=32,
            env_num=8,
        ),
        eval=dict(env_num=1, evaluator=dict(eval_freq=100, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.05,
                decay=1e5,
            ),
            replay_buffer=dict(replay_buffer_size=50000, ),
        ),
    ),
)
memory_len_r2d2_config = EasyDict(memory_len_r2d2_config)
main_config = memory_len_r2d2_config
memory_len_r2d2_create_config = dict(
    env=dict(
        type='bsuite',
        import_names=['dizoo.bsuite.envs.bsuite_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='r2d2'),
)
memory_len_r2d2_create_config = EasyDict(memory_len_r2d2_create_config)
create_config = memory_len_r2d2_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c memory_len_15_r2d2_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
