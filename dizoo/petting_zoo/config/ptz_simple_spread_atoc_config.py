from easydict import EasyDict

n_agent = 4
n_landmark = n_agent  # In simple_spread_v2, n_landmark must = n_agent
collector_env_num = 4
evaluator_env_num = 2
communication = True
ptz_simple_spread_atoc_config = dict(
    env=dict(
        env_family='mpe',
        env_id='simple_spread_v2',
        n_agent=n_agent,
        n_landmark=n_landmark,
        max_cycles=100,
        agent_obs_only=True,
        continuous_actions=True,
        act_scale=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=10,
        stop_value=0,
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            obs_shape=2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2,
            action_shape=5,
            n_agent=n_agent,
            communication=communication,
            thought_size=16,
            agent_per_group=min(n_agent // 2, 5),
        ),
        learn=dict(
            update_per_collect=5,
            batch_size=32,
            learning_rate_actor=0.001,
            learning_rate_critic=0.001,
            ignore_done=True,
            target_theta=0.005,
            discount_factor=0.9,
            communication=communication,
            actor_update_freq=1,
            noise=True,
            noise_sigma=0.15,
            noise_range=dict(
                min=-0.5,
                max=0.5,
            ),
        ),
        collect=dict(
            n_sample=500,
            unroll_len=1,
            noise_sigma=0.4,
        ),
        other=dict(replay_buffer=dict(replay_buffer_size=100000, ), )
    ),
)
ptz_simple_spread_atoc_config = EasyDict(ptz_simple_spread_atoc_config)
main_config = ptz_simple_spread_atoc_config
ptz_simple_spread_atoc_create_config = dict(
    env=dict(
        import_names=['dizoo.petting_zoo.envs.petting_zoo_env'],
        type='petting_zoo',
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='atoc'),
)
ptz_simple_spread_atoc_create_config = EasyDict(ptz_simple_spread_atoc_create_config)
create_config = ptz_simple_spread_atoc_create_config
