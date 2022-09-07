from easydict import EasyDict

spaceinvaders_r2d2_gtrxl_config = dict(
    exp_name='spaceinvaders_r2d2_gtrxl_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=10000000000,
        env_id='SpaceInvaders-v4',
        #'ALE/SpaceInvaders-v5' is available. But special setting is needed after gym make.
        frame_stack=4,
        manager=dict(shared_memory=False)
    ),
    policy=dict(
        cuda=True,
        priority=True,
        priority_IS_weight=True,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            hidden_size=2048,
            encoder_hidden_size_list=[128, 512, 2048],
            gru_bias=1.0,
            memory_len=0,
            dropout=0.2,
            att_layer_num=5,
            att_head_dim=512,
        ),
        discount_factor=0.99,
        nstep=3,
        burnin_step=0,
        unroll_len=13,
        seq_len=10,
        learn=dict(
            update_per_collect=8,
            batch_size=64,
            learning_rate=0.0005,
            target_update_theta=0.001,
            value_rescale=True,
            init_memory='zero',
        ),
        collect=dict(
            # NOTE: It is important that set key traj_len_inf=True here,
            # to make sure self._traj_len=INF in serial_sample_collector.py.
            # In sequence-based policy, for each collect_env,
            # we want to collect data of length self._traj_len=INF
            # unless the episode enters the 'done' state.
            # In each collect phase, we collect a total of <n_sample> sequence samples.
            n_sample=32,
            traj_len_inf=True,
            env_num=8,
        ),
        eval=dict(env_num=8, ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.05,
                decay=1e5,
            ),
            replay_buffer=dict(
                replay_buffer_size=10000,
                # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
                alpha=0.6,
                # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
                beta=0.4,
            )
        ),
    ),
)
spaceinvaders_r2d2_gtrxl_config = EasyDict(spaceinvaders_r2d2_gtrxl_config)
main_config = spaceinvaders_r2d2_gtrxl_config
spaceinvaders_r2d2_gtrxl_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='r2d2_gtrxl'),
)
spaceinvaders_r2d2_gtrxl_create_config = EasyDict(spaceinvaders_r2d2_gtrxl_create_config)
create_config = spaceinvaders_r2d2_gtrxl_create_config

if __name__ == '__main__':
    # or you can enter ding -m serial -c spaceinvaders_r2d2_gtrxl_config.py -s 0
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
