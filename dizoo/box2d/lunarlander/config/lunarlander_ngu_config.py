from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 8
nstep = 5
lunarlander_ngu_config = dict(
    exp_name='lunarlander_ngu_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        env_id='LunarLander-v2',
        ObsPlusPrevActRewWrapper=True,  # specific env wrapper for ngu policy
        n_evaluator_episode=evaluator_env_num,
        stop_value=195,
    ),
    rnd_reward_model=dict(
        intrinsic_reward_type='add',
        learning_rate=5e-4,
        obs_shape=8,
        action_shape=4,
        batch_size=320,  # transitions
        update_per_collect=10,
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
        obs_shape=8,
        action_shape=4,
        batch_size=320,  # transitions
        update_per_collect=10,
        only_use_last_five_frames_for_icm_rnd=False,
        clear_buffer_per_iters=10,
        nstep=nstep,
        hidden_size_list=[128, 128, 64],
        type='episodic',
    ),
    policy=dict(
        cuda=True,
        priority=True,
        priority_IS_weight=True,
        discount_factor=0.997,
        burnin_step=2,
        nstep=nstep,
        # (int) the whole sequence length to unroll the RNN network minus
        # the timesteps of burnin part,
        # i.e., <the whole sequence length> = <burnin_step> + <unroll_len>
        unroll_len=98,  # set this key according to the episode length
        model=dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[128, 128, 64],
            collector_env_num=collector_env_num,
        ),
        learn=dict(
            # according to the R2D2 paper, actor parameter update interval is 400
            # environment timesteps, and in per collect phase, we collect 32 sequence
            # samples, the length of each samlpe sequence is <burnin_step> + <unroll_len>,
            # which is 100 in our seeting, 32*100/400=8, so we set update_per_collect=8
            # in most environments
            update_per_collect=8,
            batch_size=32,
            learning_rate=1e-4,
            target_update_theta=0.001,
        ),
        collect=dict(
            # NOTE: It is important that don't include key <n_sample> here,
            # to make sure self._traj_len=INF in serial_sample_collector.py.
            # In R2D2 policy, for each collect_env, we want to collect data of length self._traj_len=INF
            # unless the episode enters the 'done' state.
            # In each collect phase, we collect a total of <n_sequence_sample> sequence samples.
            n_sequence_sample=320,
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
            replay_buffer=dict(replay_buffer_size=int(5e4),
                               # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
                               alpha=0.6,
                               # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
                               beta=0.4,
                               )
        ),
    ),
)
lunarlander_ngu_config = EasyDict(lunarlander_ngu_config)
main_config = lunarlander_ngu_config
lunarlander_ngu_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    # TODO(pu): How to be compatible with subprocess env manager when we use ObsPlusPrevActRewWrapper
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(type='ngu'),
    rnd_reward_model=dict(type='rnd-ngu'),
    episodic_reward_model=dict(type='episodic'),
    collector=dict(type='sample_ngu',)
)
lunarlander_ngu_create_config = EasyDict(lunarlander_ngu_create_config)
create_config = lunarlander_ngu_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_ngu -c lunarlander_ngu_config.py -s 0`
    from ding.entry import serial_pipeline_ngu
    serial_pipeline_ngu([main_config, create_config], seed=0)
