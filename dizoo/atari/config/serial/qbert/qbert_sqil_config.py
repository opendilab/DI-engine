from easydict import EasyDict

qbert_sqil_config = dict(
    exp_name='qbert_sqil_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=30000,
        env_id='QbertNoFrameskip-v4',
        frame_stack=4,
        manager=dict(shared_memory=False, reset_inplace=True)
    ),
    policy=dict(
        cuda=True,
        priority=True,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        nstep=3,
        discount_factor=0.97,  # discount_factor: 0.97-0.99
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=500,
            alpha=0.1  # alpha: 0.08-0.12
        ),
        collect=dict(n_sample=100,
                     # Users should add their own model path here. Model path should lead to a model.
                     # Absolute path is recommended.
                     # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
                     demonstration_info_path='model_path'
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
qbert_sqil_config = EasyDict(qbert_sqil_config)
main_config = qbert_sqil_config
qbert_sqil_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
qbert_sqil_create_config = EasyDict(qbert_sqil_create_config)
create_config = qbert_sqil_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial_sqil -c qbert_sqil_config.py -s 0`
    # then input the config you used to generate your expert model in the path mentioned above
    # e.g. qbert_dqn_config.py
    from ding.entry import serial_pipeline_sqil
    from dizoo.atari.config.serial.qbert import qbert_dqn_config, qbert_dqn_create_config
    expert_main_config = qbert_dqn_config
    expert_create_config = qbert_dqn_create_config
    serial_pipeline_sqil([main_config, create_config], [expert_main_config, expert_create_config], seed=0)
