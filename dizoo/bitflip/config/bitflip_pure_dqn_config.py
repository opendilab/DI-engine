from easydict import EasyDict

n_bits = 4
bitflip_pure_dqn_config = dict(
    exp_name='bitflip_{}bit_puredqn_seed0'.format(n_bits),
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_bits=n_bits,
        n_evaluator_episode=8,
        stop_value=0.9,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=2 * n_bits,
            action_shape=n_bits,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        discount_factor=0.9,
        learn=dict(
            update_per_collect=10,
            batch_size=128,
            learning_rate=0.0005,
            target_update_freq=500,
        ),
        collect=dict(n_episode=8, unroll_len=1, collector=dict(get_train_sample=True, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=4000, ),
        ),
    ),
)
bitflip_pure_dqn_config = EasyDict(bitflip_pure_dqn_config)
main_config = bitflip_pure_dqn_config

bitflip_pure_dqn_create_config = dict(
    env=dict(
        type='bitflip',
        import_names=['dizoo.bitflip.envs.bitflip_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
    replay_buffer=dict(type='episode'),
    collector=dict(type='episode'),
)
bitflip_pure_dqn_create_config = EasyDict(bitflip_pure_dqn_create_config)
create_config = bitflip_pure_dqn_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c bitflip_pure_dqn_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
