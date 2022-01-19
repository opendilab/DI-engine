from ding.policy import PolicyFactory


def random_collect_fn(policy_cfg, policy, collector, collector_env, commander, replay_buffer, mark_not_expert=False, warm_up=False):
    assert policy_cfg.random_collect_size > 0
    if policy_cfg.get('transition_with_policy_data', False):
        collector.reset_policy(policy.collect_mode)
    else:
        action_space = collector_env.env_info().act_space
        random_policy = PolicyFactory.get_random_policy(policy.collect_mode, action_space=action_space)
        collector.reset_policy(random_policy)
    collect_kwargs = commander.step()
    # if hasattr(collector, '_default_n_episode'):
    if policy_cfg.collect.collector.type == 'episode':
        new_data = collector.collect(n_episode=policy_cfg.random_collect_size, policy_kwargs=collect_kwargs)
    else:
        new_data = collector.collect(n_sample=policy_cfg.random_collect_size, policy_kwargs=collect_kwargs)
    if mark_not_expert:
        for i in range(len(new_data)):
            new_data[i]['is_expert'] = 0  # set is_expert flag(expert 1, agent 0)
    if warm_up:
        # for td3_vae
        for i in range(len(new_data)):
            new_data[i]['warm_up'] = True
    replay_buffer.push(new_data, cur_collector_envstep=0)
    collector.reset_policy(policy.collect_mode)
