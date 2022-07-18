from easydict import EasyDict

lunarlander_c51_config = dict(
    exp_name='lunarlander_c51_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        env_id='LunarLander-v2',
        n_evaluator_episode=8,
        stop_value=200,
    ),
    policy=dict(
        cuda=False,
        priority=True,
        model=dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[128, 128, 64],
            v_min=-10,
            v_max=10,
            n_atom=51,
        ),
        discount_factor=0.97,
        nstep=3,
        learn=dict(
            update_per_collect=3,
            batch_size=64,
            learning_rate=0.001,
            target_update_freq=100,
        ),
        collect=dict(
            n_sample=80,
            unroll_len=1,
        ),
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
lunarlander_c51_config = EasyDict(lunarlander_c51_config)
main_config = lunarlander_c51_config
lunarlander_c51_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='c51'),
)
lunarlander_c51_create_config = EasyDict(lunarlander_c51_create_config)
create_config = lunarlander_c51_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c lunarlander_c51_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
