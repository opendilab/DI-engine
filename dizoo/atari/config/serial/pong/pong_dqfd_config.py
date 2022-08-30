from easydict import EasyDict

pong_dqfd_config = dict(
    exp_name='pong_dqfd_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=20,
        env_id='Pong-v4',
        #'ALE/Pong-v5' is available. But special setting is needed after gym make.
        frame_stack=4,
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
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=500,
            lambda1=1.0,
            lambda2=1.0,
            lambda3=1e-5,
            per_train_iter_k=10,
            expert_replay_buffer_size=10000,
            # justify the buffer size of the expert buffer
        ),
        collect=dict(
            n_sample=64,
            # Users should add their own model path here. Model path should lead to a model.
            # Absolute path is recommended.
            # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
            model_path='model_path_placeholder',
            # Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        # Users should add their own path here (path should lead to a well-trained model)
        # Absolute path is recommended
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=250000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, ),
        ),
    ),
)
pong_dqfd_config = EasyDict(pong_dqfd_config)
main_config = pong_dqfd_config
pong_dqfd_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqfd'),
)
pong_dqfd_create_config = EasyDict(pong_dqfd_create_config)
create_config = pong_dqfd_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial_dqfd -c pong_dqfd_config.py -s 0`
    # then input ``pong_dqfd_config.py`` upon the instructions.
    # The reason we need to input the dqfd config is we have to borrow its ``_get_train_sample`` function
    # in the collector part even though the expert model may be generated from other Q learning algos.
    from ding.entry.serial_entry_dqfd import serial_pipeline_dqfd
    from dizoo.atari.config.serial.pong import pong_dqfd_config, pong_dqfd_create_config
    expert_main_config = pong_dqfd_config
    expert_create_config = pong_dqfd_create_config
    serial_pipeline_dqfd((main_config, create_config), (expert_main_config, expert_create_config), seed=0)
