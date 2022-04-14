from easydict import EasyDict

memory_len_dqn_config = dict(
    exp_name='memory_len_0_dqn_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=1,
        n_evaluator_episode=20,
        env_id='memory_len/0',  # this environment configuration is 1 'memory steps' long
        stop_value=1.,
        manager=dict(shared_memory=False, )
    ),
    policy=dict(
        cuda=True,
        priority=True,
        model=dict(
            obs_shape=3,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        nstep=1,
        discount_factor=0.97,  # discount_factor: 0.97-0.99
        learn=dict(
            batch_size=64,
            learning_rate=0.001,
        ),
        collect=dict(n_sample=8),
        eval=dict(evaluator=dict(eval_freq=100, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=20000, ),
        ),
    ),
)
memory_len_dqn_config = EasyDict(memory_len_dqn_config)
main_config = memory_len_dqn_config
memory_len_dqn_create_config = dict(
    env=dict(
        type='bsuite',
        import_names=['dizoo.bsuite.envs.bsuite_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
memory_len_dqn_create_config = EasyDict(memory_len_dqn_create_config)
create_config = memory_len_dqn_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c memory_len_0_dqn_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
