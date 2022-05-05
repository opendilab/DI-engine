from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 8
lunarlander_r2d2_gtrxl_config = dict(
    exp_name='lunarlander_r2d2_gtrxl_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=8,
        stop_value=200,
        env_id='LunarLander-v2',
    ),
    policy=dict(
        cuda=True,
        priority=True,
        priority_IS_weight=True,
        model=dict(
            obs_shape=8,
            action_shape=4,
            memory_len=0,  # length of transformer memory (can be 0)
            hidden_size=256,
            gru_bias=1.,
            att_layer_num=3,
            dropout=0.1,
            att_head_dim=64,
            att_head_num=8,
        ),
        discount_factor=0.99,
        nstep=5,
        burnin_step=0,  # how many steps use to initialize the memory (can be 0)
        unroll_len=25,  # trajectory len
        seq_len=20,  # transformer input segment
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
            # In sequence-based policy, for each collect_env,
            # we want to collect data of length self._traj_len=INF
            # unless the episode enters the 'done' state.
            # In each collect phase, we collect a total of <n_sample> sequence samples.
            n_sample=32,
            traj_len_inf=True,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.05,
                decay=1e5,
            ),
            replay_buffer=dict(
                replay_buffer_size=50000,
                # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
                alpha=0.6,
                # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
                beta=0.4,
            )
        ),
    ),
)
lunarlander_r2d2_gtrxl_config = EasyDict(lunarlander_r2d2_gtrxl_config)
main_config = lunarlander_r2d2_gtrxl_config
lunarlander_r2d2_gtrxl_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='r2d2_gtrxl'),
)
lunarlander_r2d2_gtrxl_create_config = EasyDict(lunarlander_r2d2_gtrxl_create_config)
create_config = lunarlander_r2d2_gtrxl_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c lunarlander_r2d2_gtrxl_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)
