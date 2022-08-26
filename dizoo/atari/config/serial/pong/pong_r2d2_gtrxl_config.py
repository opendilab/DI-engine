from easydict import EasyDict

collector_env_num = 4
evaluator_env_num = 4
pong_r2d2_gtrxl_config = dict(
    exp_name='pong_r2d2_gtrxl_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=5,
        stop_value=20,
        env_id='Pong-v4',
        #'ALE/Pong-v5' is available. But special setting is needed after gym make.
        frame_stack=4,
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        priority=True,
        priority_IS_weight=True,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            hidden_size=1024,
            encoder_hidden_size_list=[128, 512, 1024],
            gru_bias=2.,
            memory_len=0,
            dropout=0.1,
            att_head_num=8,
            att_layer_num=3,
            att_head_dim=16,
        ),
        discount_factor=0.997,
        burnin_step=0,
        nstep=5,
        unroll_len=25,
        seq_len=20,
        learn=dict(
            update_per_collect=8,
            batch_size=64,
            learning_rate=0.0005,
            target_update_theta=0.001,
            value_rescale=True,
        ),
        collect=dict(
            # NOTE: It is important that set key traj_len_inf=True here,
            # to make sure self._traj_len=INF in serial_sample_collector.py.
            # In R2D2 policy, for each collect_env, we want to collect data of length self._traj_len=INF
            # unless the episode enters the 'done' state.
            # In each collect phase, we collect a total of <n_sample> sequence samples.
            n_sample=32,
            traj_len_inf=True,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, evaluator=dict(eval_freq=300, )),
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
pong_r2d2_gtrxl_config = EasyDict(pong_r2d2_gtrxl_config)
main_config = pong_r2d2_gtrxl_config
pong_r2d2_gtrxl_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='r2d2_gtrxl'),
)
pong_r2d2_gtrxl_create_config = EasyDict(pong_r2d2_gtrxl_create_config)
create_config = pong_r2d2_gtrxl_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c pong_r2d2_gtrxl_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)
