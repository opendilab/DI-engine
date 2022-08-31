from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 8
lunarlander_r2d2_config = dict(
    exp_name='lunarlander_r2d2_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        env_id='LunarLander-v2',
        n_evaluator_episode=8,
        stop_value=200,
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        priority=True,
        priority_IS_weight=True,
        model=dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        discount_factor=0.997,
        burnin_step=2,
        nstep=5,
        # (int) the whole sequence length to unroll the RNN network minus
        # the timesteps of burnin part,
        # i.e., <the whole sequence length> = <unroll_len> = <burnin_step> + <learn_unroll_len>
        learn_unroll_len=40,
        learn=dict(
            # according to the R2D2 paper, actor parameter update interval is 400
            # environment timesteps, and in per collect phase, we collect <n_sample> sequence
            # samples, the length of each sequence sample is <burnin_step> + <learn_unroll_len>,
            # e.g. if  n_sample=32, <sequence length> is 100, thus 32*100/400=8,
            # we will set update_per_collect=8 in most environments.
            update_per_collect=8,
            batch_size=64,
            learning_rate=0.0005,
            target_update_theta=0.001,
        ),
        collect=dict(
            # NOTE: It is important that set key traj_len_inf=True here,
            # to make sure self._traj_len=INF in serial_sample_collector.py.
            # In R2D2 policy, for each collect_env, we want to collect data of length self._traj_len=INF
            # unless the episode enters the 'done' state.
            # In each collect phase, we collect a total of <n_sample> sequence samples.
            n_sample=32,
            unroll_len=2 + 40,
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
lunarlander_r2d2_config = EasyDict(lunarlander_r2d2_config)
main_config = lunarlander_r2d2_config
lunarlander_r2d2_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='r2d2'),
)
lunarlander_r2d2_create_config = EasyDict(lunarlander_r2d2_create_config)
create_config = lunarlander_r2d2_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c lunarlander_r2d2_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)
