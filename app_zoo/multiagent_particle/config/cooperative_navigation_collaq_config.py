from easydict import EasyDict

agent_num = 5
num_agents = agent_num
num_landmarks = agent_num
collector_env_num = 4
evaluator_env_num = 2
cooperative_navigation_collaq_config = dict(
    env=dict(
        num_agents=num_agents,
        num_landmarks=num_landmarks,
        max_step=100,
        agent_num=agent_num,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        manager=dict(
            shared_memory=False,
        ),
    ),
    policy=dict(
        cuda=True,
        on_policy=True,
        model=dict(
            agent_num=agent_num,
            obs_shape=2 + 2 + (agent_num - 1) * 2 + num_landmarks * 2,
            alone_obs_shape=2 + 2 + (num_landmarks) * 2,
            global_obs_shape=agent_num * 2 + num_landmarks * 2 + agent_num * 2,
            action_shape=5,
            hidden_size_list=[128, 128, 64],
            attention=True,
            self_feature_range=[2, 4],  # placeholder
            ally_feature_range=[4, agent_num * 2 + 2],  # placeholder
            attention_size=32,
        ),
        learn=dict(
            update_per_collect=100,
            batch_size=32,
            agent_num=agent_num,
            learning_rate=0.0001,
            target_update_theta=0.001,
            discount_factor=0.99,
        ),
        collect=dict(
            n_episode=6,
            unroll_len=16,
            agent_num=agent_num,
            env_num=collector_env_num,
        ),
        eval=dict(
            agent_num=agent_num,
            env_num=evaluator_env_num,
        ),
        other=dict(eps=dict(
            type='exp',
            start=1.0,
            end=0.05,
            decay=100000,
        ), ),
    ),
)
cooperative_navigation_collaq_config = EasyDict(cooperative_navigation_collaq_config)
main_config = cooperative_navigation_collaq_config
cooperative_navigation_collaq_create_config = dict(
    env=dict(
        import_names=['app_zoo.multiagent_particle.envs.particle_env'],
        type='cooperative_navigation',
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='collaq'),
)
cooperative_navigation_collaq_create_config = EasyDict(cooperative_navigation_collaq_create_config)
create_config = cooperative_navigation_collaq_create_config
