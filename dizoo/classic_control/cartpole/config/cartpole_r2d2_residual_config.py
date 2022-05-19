from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 8
cartpole_r2d2__residual_config = dict(
    exp_name='cartpole_r2d2_residual_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        priority=False,
        priority_IS_weight=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            res_link=True,
        ),
        discount_factor=0.997,
        nstep=5,
        burnin_step=10,
        # (int) <learn_unroll_len> is the total length of [sequence sample] minus
        # the length of burnin part in [sequence sample],
        # i.e., <sequence sample length> = <unroll_len> = <burnin_step> + <learn_unroll_len>
        learn_unroll_len=20,  # set this key according to the episode length
        learn=dict(
            # according to the R2D2 paper, actor parameter update interval is 400
            # environment timesteps, and in per collect phase, we collect <n_sample> sequence
            # samples, the length of each sequence sample is <burnin_step> + <learn_unroll_len>,
            # e.g. if n_sample=32, <sequence length> is 100, thus 32*100/400=8,
            # we will set update_per_collect=8 in most environments.
            update_per_collect=8,
            batch_size=64,
            learning_rate=0.0005,
            # according to the R2D2 paper, the target network update interval is 2500
            target_update_freq=2500,
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
        eval=dict(env_num=evaluator_env_num, evaluator=dict(eval_freq=20)),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.05,
                decay=10000,
            ), replay_buffer=dict(replay_buffer_size=100000, )
        ),
    ),
)
cartpole_r2d2__residual_config = EasyDict(cartpole_r2d2__residual_config)
main_config = cartpole_r2d2__residual_config
cartpole_r2d2_residual_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='r2d2'),
)
cartpole_r2d2_residual_create_config = EasyDict(cartpole_r2d2_residual_create_config)
create_config = cartpole_r2d2_residual_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c cartpole_r2d2_residual_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
