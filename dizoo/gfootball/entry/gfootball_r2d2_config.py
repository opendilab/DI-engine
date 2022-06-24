from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 5
# debug
# collector_env_num = 1
# evaluator_env_num = 1  
gfootball_r2d2_main_config = dict(
    exp_name='data_gfootball/gfootball_easy_r2d2_seed0',
    # exp_name='data_gfootball/gfootball_medium_r2d2_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=999,
        # env_name="11_vs_11_stochastic",  # default: medium
        env_name="11_vs_11_easy_stochastic",
    ),
    policy=dict(
        cuda=True,
        priority=True,
        priority_IS_weight=True,
        nstep=5,
        discount_factor=0.997,
        burnin_step=20,
        # (int) the whole sequence length to unroll the RNN network minus
        # the timesteps of burnin part,
        # i.e., <the whole sequence length> = <unroll_len> = <burnin_step> + <learn_unroll_len>
        learn_unroll_len=80,
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
            traj_len_inf=True,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, ),
        other=dict(
            eps=dict(
                type='exp',
                start=1,
                end=0.05,
                decay=1e5,
            ),
            replay_buffer=dict(
                replay_buffer_size=int(1e4),
                # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
                alpha=0.6,
                # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
                beta=0.4,
            )
        ),
    ),
)
gfootball_r2d2_main_config = EasyDict(gfootball_r2d2_main_config)
main_config = gfootball_r2d2_main_config

gfootball_r2d2_create_config = dict(
    env=dict(
        type='gfootball',
        import_names=['dizoo.gfootball.envs.gfootball_env'],
    ),
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(type='r2d2'),
)
gfootball_r2d2_create_config = EasyDict(gfootball_r2d2_create_config)
create_config = gfootball_r2d2_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c gfootball_r2d2_config.py -s 0`
    from ding.entry import serial_pipeline
    from dizoo.gfootball.model.q_network.football_q_network import FootballDRQN
    football_drqn = FootballDRQN(env_name='football')
    serial_pipeline((main_config, create_config), model=football_drqn, seed=0, max_env_step=10e6)
