from easydict import EasyDict

# debug
# collector_env_num = 2
# evaluator_env_num = 2

collector_env_num = 5
evaluator_env_num = 5

nstep = 5

gfootball_ngu_main_config = dict(
    exp_name='data_gfootball/gfootball_easy_ngu_seed0_rbs1e3_bs32',
    # exp_name='data_gfootball/gfootball_medium_ngu_seed0_rbs1e3_bs32',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=999,
        # env_name="11_vs_11_hard_stochastic",
        # env_name="11_vs_11_stochastic",  # default: medium
        env_name="11_vs_11_easy_stochastic",
        save_replay_gif=False,
        obs_plus_prev_action_reward=True,  # use specific env wrapper for ngu policy
    ),
    rnd_reward_model=dict(
        intrinsic_reward_type='add',
        learning_rate=5e-4,
        obs_shape=1312,
        action_shape=19,
        batch_size=320,  # transitions
        update_per_collect=10,  # 32*100/320=10
        only_use_last_five_frames_for_icm_rnd=False,
        clear_buffer_per_iters=10,
        nstep=nstep,
        hidden_size_list=[128, 128, 64],
        type='rnd-ngu',
    ),
    episodic_reward_model=dict(
        # means if using rescale trick to the last non-zero reward
        # when combing extrinsic and intrinsic reward.
        # the rescale trick only used in:
        # 1. sparse reward env minigrid, in which the last non-zero reward is a strong positive signal
        # 2. the last reward of each episode directly reflects the agent's completion of the task, e.g. lunarlander
        # Note that the ngu intrinsic reward is a positive value (max value is 5), in these envs,
        # the last non-zero reward should not be overwhelmed by intrinsic rewards, so we need rescale the
        # original last nonzero extrinsic reward.
        # please refer to ngu_reward_model for details.
        last_nonzero_reward_rescale=True,
        # means the rescale value for the last non-zero reward, only used when last_nonzero_reward_rescale is True
        # please refer to ngu_reward_model for details.
        last_nonzero_reward_weight=100,
        intrinsic_reward_type='add',
        learning_rate=5e-4,
        obs_shape=1312,
        action_shape=19,
        batch_size=320,  # transitions
        update_per_collect=10,  # 32*100/64=50
        only_use_last_five_frames_for_icm_rnd=False,
        clear_buffer_per_iters=10,
        nstep=nstep,
        hidden_size_list=[128, 128, 64],
        type='episodic',
    ),
    policy=dict(
        il_model_path=None,
        rl_model_path=None,
        replay_path=None,
        env_name="football",
        cuda=True,
        priority=True,
        priority_IS_weight=True,
        nstep=nstep,
        discount_factor=0.997,
        burnin_step=20,
        # (int) the whole sequence length to unroll the RNN network minus
        # the timesteps of burnin part,
        # i.e., <the whole sequence length> = <unroll_len> = <burnin_step> + <learn_unroll_len>
        learn_unroll_len=80,  # set this key according to the episode length
        learn=dict(
            # according to the ngu paper, actor parameter update interval is 400
            # environment timesteps, and in per collect phase, we collect <n_sample> sequence
            # samples, the length of each sequence sample is <burnin_step> + <learn_unroll_len>,
            # e.g. if  n_sample=32, <sequence length> is 100, thus 32*100/400=8,
            # we will set update_per_collect=8 in most environments.
            # debug
            # update_per_collect=2,
            # batch_size=4,
            update_per_collect=16,
            batch_size=32,
            learning_rate=0.0005,
            target_update_theta=0.001,
        ),
        collect=dict(
            # NOTE: It is important that set key traj_len_inf=True here,
            # to make sure self._traj_len=INF in serial_sample_collector.py.
            # In ngu policy, for each collect_env, we want to collect data of length self._traj_len=INF
            # unless the episode enters the 'done' state.
            # In each collect phase, we collect a total of <n_sample> sequence samples.
            # debug
            # n_sample=2,
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
                replay_buffer_size=int(1e3),
                # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
                alpha=0.6,
                # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
                beta=0.4,
            )
        ),
    ),
)
gfootball_ngu_main_config = EasyDict(gfootball_ngu_main_config)
main_config = gfootball_ngu_main_config

gfootball_ngu_create_config = dict(
    env=dict(
        type='gfootball',
        import_names=['dizoo.gfootball.envs.gfootball_env'],
    ),
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(type='ngu'),
    rnd_reward_model=dict(type='rnd-ngu'),
    episodic_reward_model=dict(type='episodic'),
)
gfootball_ngu_create_config = EasyDict(gfootball_ngu_create_config)
create_config = gfootball_ngu_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c gfootball_ngu_config.py -s 0`
    from ding.entry import serial_pipeline_ngu_gfootball
    from dizoo.gfootball.model.q_network.football_q_network import FootballNGU
    football_ngu = FootballNGU(env_name='football')
    serial_pipeline_ngu_gfootball((main_config, create_config), model=football_ngu, seed=0, max_env_step=10e6)
