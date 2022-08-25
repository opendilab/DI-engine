from easydict import EasyDict

qbert_cql_config = dict(
    exp_name='qbert_cql_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=30000,
        env_id='Qbert-v4',
        #'ALE/Qbert-v5' is available. But special setting is needed after gym make.
        frame_stack=4
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
            num_quantiles=200,
        ),
        nstep=1,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            train_epoch=30000,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=2000,
            min_q_weight=10.0,
        ),
        collect=dict(
            n_sample=100,
            data_type='naive',
            # Users should add their own data path here. Data path should lead to a file to store data or load the stored data.
            # Absolute path is recommended.
            # In DI-engine, it is usually located in ``exp_name`` directory
            data_path='data_path_placeholder',
        ),
        eval=dict(evaluator=dict(eval_freq=4000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=1000000,
            ),
            replay_buffer=dict(replay_buffer_size=400000, ),
        ),
    ),
)
main_config = EasyDict(qbert_cql_config)

qbert_cql_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='cql_discrete'),
)
create_config = EasyDict(qbert_cql_create_config)

if __name__ == '__main__':
    # or you can enter ding -m serial_offline -c qbert_cql_config.py -s 0
    from ding.entry import serial_pipeline_offline
    serial_pipeline_offline((main_config, create_config), seed=0)
