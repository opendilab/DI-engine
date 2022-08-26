from easydict import EasyDict

qbert_qrdqn_config = dict(
    exp_name='qbert_qrdqn_generation_data_seed0',
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
            num_quantiles=64,
        ),
        nstep=1,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=500,
            learner=dict(
                load_path='./expert/ckpt/ckpt_best.pth.tar',
                hook=dict(
                    load_ckpt_before_run='./expert/ckpt/ckpt_best.pth.tar',
                    save_ckpt_after_run=False,
                )
            ),
        ),
        collect=dict(
            n_sample=100,
            data_type='hdf5',
            save_path='./expert/expert.pkl',
        ),
        eval=dict(evaluator=dict(eval_freq=4000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=1000000,
                collect=0.2,
            ),
            replay_buffer=dict(replay_buffer_size=100000, ),
        ),
    ),
)
main_config = EasyDict(qbert_qrdqn_config)

qbert_qrdqn_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='qrdqn'),
)
create_config = EasyDict(qbert_qrdqn_create_config)

if __name__ == '__main__':
    # or you can enter ding -m serial -c qbert_qrdqn_generation_data_config.py -s 0
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
