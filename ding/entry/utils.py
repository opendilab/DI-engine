from typing import Optional, Callable, List

from ding.policy import PolicyFactory


def mark_not_expert(ori_data: List[dict]) -> List[dict]:
    for i in range(len(ori_data)):
        # Set is_expert flag (expert 1, agent 0)
        ori_data[i]['is_expert'] = 0
    return ori_data


def mark_warm_up(ori_data: List[dict]) -> List[dict]:
    # for td3_vae
    for i in range(len(ori_data)):
        ori_data[i]['warm_up'] = True
    return ori_data


def random_collect(
        policy_cfg: 'EasyDict',  # noqa
        policy: 'Policy',  # noqa
        collector: 'ISerialCollector',  # noqa
        collector_env: 'BaseEnvManager',  # noqa
        commander: 'BaseSerialCommander',  # noqa
        replay_buffer: 'IBuffer',  # noqa
        postprocess_data_fn: Optional[Callable] = None
) -> None:  # noqa
    assert policy_cfg.random_collect_size > 0
    if policy_cfg.get('transition_with_policy_data', False):
        collector.reset_policy(policy.collect_mode)
    else:
        action_space = collector_env.action_space
        random_policy = PolicyFactory.get_random_policy(policy.collect_mode, action_space=action_space)
        collector.reset_policy(random_policy)
    collect_kwargs = commander.step()
    if policy_cfg.collect.collector.type == 'episode':
        new_data = collector.collect(n_episode=policy_cfg.random_collect_size, policy_kwargs=collect_kwargs)
    else:
        new_data = collector.collect(n_sample=policy_cfg.random_collect_size, policy_kwargs=collect_kwargs)
    if postprocess_data_fn is not None:
        new_data = postprocess_data_fn(new_data)
    replay_buffer.push(new_data, cur_collector_envstep=0)
    collector.reset_policy(policy.collect_mode)
