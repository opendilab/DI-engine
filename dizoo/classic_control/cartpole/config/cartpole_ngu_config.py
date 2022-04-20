from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 5
nstep = 5
cartpole_ngu_config = dict(
    exp_name='cartpole_ngu_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=5,
        stop_value=195,
        # replay_path='eval_replay', #TODO
    ),
    rnd_reward_model=dict(
        intrinsic_reward_type='add',  # 'assign'
        learning_rate=0.001,
        obs_shape=4,
        action_shape=2,
        batch_size=64,
        update_per_collect=50,  # 32*100/64=50
        clear_buffer_per_iters=10,
        nstep=nstep,
        hidden_size_list=[128, 128, 64],
        type='rnd',
    ),
    episodic_reward_model=dict(
        intrinsic_reward_type='add',
        learning_rate=0.001,
        obs_shape=4,
        action_shape=2,
        batch_size=64,
        update_per_collect=50,
        nstep=nstep,
        hidden_size_list=[128, 128, 64],
        type='episodic',
    ),
    policy=dict(
        cuda=False,
        on_policy=False,
        priority=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            collector_env_num=collector_env_num,  # TODO(pu)
        ),
        discount_factor=0.997,
        burnin_step=20,
        nstep=nstep,
        # (int) the whole sequence length to unroll the RNN network minus
        # the timesteps of burnin part,
        # i.e., <the whole sequence length> = <burnin_step> + <unroll_len>
        unroll_len=80,
        learn=dict(
            # according to the R2D2 paper, actor parameter update interval is 400
            # environment timesteps, and in per collect phase, we collect 32 sequence
            # samples, the length of each samlpe sequence is <burnin_step> + <unroll_len>,
            # which is 100 in our seeting, 32*100/400=8, so we set update_per_collect=8
            # in most environments
            update_per_collect=8,
            batch_size=64,
            learning_rate=0.0005,
            # according to the R2D2 paper, the target network update interval is 2500
            target_update_freq=2500,
        ),
        collect=dict(
            n_sample=32,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, ),
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
cartpole_ngu_config = EasyDict(cartpole_ngu_config)
main_config = cartpole_ngu_config
cartpole_ngu_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ngu'),
    rnd_reward_model=dict(type='rnd'),
    episodic_reward_model=dict(type='episodic'),
    collector=dict(type='sample_ngu', )
)
cartpole_ngu_create_config = EasyDict(cartpole_ngu_create_config)
create_config = cartpole_ngu_create_config

if __name__ == "__main__":
    # TODO: confirm which mode to be used in CLI
    from ding.entry import serial_pipeline_ngu

    serial_pipeline_ngu([main_config, create_config], seed=0)
