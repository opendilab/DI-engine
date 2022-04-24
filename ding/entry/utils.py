from typing import Optional, Callable, List, Any

from ding.policy import PolicyFactory
from ding.worker import IMetric, MetricSerialEvaluator


class AccMetric(IMetric):

    def eval(self, inputs: Any, label: Any) -> dict:
        return {'Acc': (inputs['logit'].sum(dim=1) == label).sum().item() / label.shape[0]}

    def reduce_mean(self, inputs: List[Any]) -> Any:
        s = 0
        for item in inputs:
            s += item['Acc']
        return {'Acc': s / len(inputs)}

    def gt(self, metric1: Any, metric2: Any) -> bool:
        if metric2 is None:
            return True
        if isinstance(metric2, dict):
            m2 = metric2['Acc']
        else:
            m2 = metric2
        return metric1['Acc'] > m2


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
