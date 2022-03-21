from easydict import EasyDict

n_agent = 5
n_landmark = n_agent
collector_env_num = 4
evaluator_env_num = 2
ptz_simple_spread_coma_config = dict(
    env=dict(
        env_family='mpe',
        env_id='simple_spread_v2',
        n_agent=n_agent,
        n_landmark=n_landmark,
        max_cycles=100,
        agent_obs_only=False,
        continuous_actions=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=5,
        stop_value=0,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        cuda=False,
        model=dict(
            agent_num=n_agent,
            obs_shape=dict(
                agent_state=2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2,
                global_state=n_agent * 4 + n_landmark * 2 + n_agent * (n_agent - 1) * 2,
            ),
            action_shape=[
                5,
            ],
            actor_hidden_size_list=[128, 128, 64],
        ),
        agent_num=n_agent,
        learn=dict(
            update_per_collect=1,
            batch_size=32,
            learning_rate=0.0005,
            target_update_theta=0.001,
            discount_factor=0.99,
            td_lambda=0.8,
            value_weight=1.0,
            entropy_weight=0.01,
        ),
        collect=dict(
            n_sample=600,
            unroll_len=16,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.5,
                end=0.01,
                decay=100000,
            ),
            replay_buffer=dict(
                replay_buffer_size=64,
                max_use=10,
            ),
        ),
    ),
)
ptz_simple_spread_coma_config = EasyDict(ptz_simple_spread_coma_config)
main_config = ptz_simple_spread_coma_config
ptz_simple_spread_coma_create_config = dict(
    env=dict(
        import_names=['dizoo.petting_zoo.envs.petting_zoo_env'],
        type='petting_zoo',
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='coma'),
)
ptz_simple_spread_coma_create_config = EasyDict(ptz_simple_spread_coma_create_config)
create_config = ptz_simple_spread_coma_create_config
