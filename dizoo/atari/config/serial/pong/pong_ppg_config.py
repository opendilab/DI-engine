from easydict import EasyDict

pong_ppg_config = dict(
    exp_name='pong_ppg_seed0',
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
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[64, 64, 128],
            critic_head_hidden_size=128,
            actor_head_hidden_size=128,
        ),
        learn=dict(
            update_per_collect=24,
            batch_size=128,
            # (bool) Whether to normalize advantage. Default to False.
            adv_norm=False,
            learning_rate=0.0001,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=0.5,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.03,
            clip_ratio=0.1,
            epochs_aux=6,
            beta_weight=1,
            aux_freq=100
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration times
            n_sample=1024,
            # (float) the trade-off factor lambda to balance 1step td and mc
            gae_lambda=0.95,
            discount_factor=0.99,
        ),
        eval=dict(evaluator=dict(eval_freq=1000, )),
        other=dict(
            replay_buffer=dict(
                multi_buffer=True,
                policy=dict(
                    replay_buffer_size=100000,
                    max_use=3,
                ),
                value=dict(
                    replay_buffer_size=100000,
                    max_use=5,
                ),
            ),
        ),
    ),
)
main_config = EasyDict(pong_ppg_config)

pong_ppg_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppg_offpolicy'),
)
create_config = EasyDict(pong_ppg_create_config)

if __name__ == "__main__":
    import os
    import warnings
    from dizoo.atari.entry.atari_ppg_main import main
    from dizoo.atari.entry.atari_ppg_main import __file__ as _origin_py_file
    origin_py_file_rel = os.path.relpath(_origin_py_file, os.path.abspath(os.path.curdir))
    warnings.warn(UserWarning(f"This config file can be executed by {repr(origin_py_file_rel)}"))
    main(main_config)
