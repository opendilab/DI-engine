from typing import Optional, List

from easydict import EasyDict

from nervex.utils import find_free_port, find_free_port_slurm, node_to_partition, node_to_host, pretty_print

default_host = '0.0.0.0'


def set_host_port(cfg: EasyDict, coordinator_host: str, learner_host: str, actor_host: str) -> EasyDict:
    cfg.coordinator.interaction.host = coordinator_host
    if cfg.coordinator.interaction.port == 'auto':
        cfg.coordinator.interaction.port = find_free_port(coordinator_host)
    learner_count = 0
    actor_count = 0
    for k in cfg.keys():
        if k.startswith('learner'):
            if cfg[k].host == 'auto':
                if isinstance(learner_host, list):
                    cfg[k].host = learner_host[learner_count]
                    learner_count += 1
                elif isinstance(learner_host, str):
                    cfg[k].host = learner_host
                else:
                    raise TypeError("not support learner_host type: {}".format(learner_host))
            if cfg[k].port == 'auto':
                cfg[k].port = find_free_port(cfg[k].host)
        if k.startswith('actor'):
            if cfg[k].host == 'auto':
                if isinstance(actor_host, list):
                    cfg[k].host = actor_host[actor_count]
                    actor_count += 1
                elif isinstance(actor_host, str):
                    cfg[k].host = actor_host
                else:
                    raise TypeError("not support actor_host type: {}".format(actor_host))
            if cfg[k].port == 'auto':
                cfg[k].port = find_free_port(cfg[k].host)
    return cfg


def set_host_port_slurm(cfg: EasyDict, coordinator_host: str, learner_node: list, actor_node: list) -> EasyDict:
    cfg.coordinator.interaction.host = coordinator_host
    if cfg.coordinator.interaction.port == 'auto':
        cfg.coordinator.interaction.port = find_free_port(coordinator_host)
    if isinstance(learner_node, str):
        learner_node = [learner_node]
    if isinstance(actor_node, str):
        actor_node = [actor_node]
    learner_count, actor_count = 0, 0
    learner_multi = {}
    for k in cfg.keys():
        if learner_node is not None and k.startswith('learner'):
            node = learner_node[learner_count % len(learner_node)]
            cfg[k].node = node
            cfg[k].partition = node_to_partition(node)
            repeat_num = cfg[k].get('repeat_num', 1)
            if cfg[k].host != 'auto':
                cfg[k].host = node_to_host(node)
            if cfg[k].port != 'auto':
                if repeat_num == 1:
                    cfg[k].port = find_free_port_slurm(node)
                    learner_multi[k] = False
                else:
                    cfg[k].port = [find_free_port_slurm(node) for _ in range(repeat_num)]
                    learner_multi[k] = True
            learner_count += 1
        if actor_node is not None and k.startswith('actor'):
            node = actor_node[actor_count % len(actor_node)]
            cfg[k].node = node
            cfg[k].partition = node_to_partition(node)
            if cfg[k].host != 'auto':
                cfg[k].host = node_to_host(node)
            if cfg[k].port != 'auto':
                cfg[k].port = find_free_port_slurm(node)
            actor_count += 1
    for k, flag in learner_multi.items():
        if flag:
            host = cfg[k].host
            learner_interaction_cfg = {str(i): [str(i), host, p] for i, p in enumerate(cfg[k].port)}
            aggregator_cfg = dict(
                master=dict(
                    host=host,
                    port=find_free_port_slurm(cfg[k].node),
                ),
                slave=dict(
                    host=host,
                    port=find_free_port_slurm(cfg[k].node),
                ),
                learner=learner_interaction_cfg,
                node=cfg[k].node,
                partition=cfg[k].partition,
            )
            cfg[k].use_aggregator = True
            cfg['aggregator' + k[7:]] = aggregator_cfg
        else:
            cfg[k].use_aggregator = False
    return cfg


def set_learner_interaction_for_coordinator(cfg: EasyDict) -> EasyDict:
    cfg.coordinator.interaction.learner = {}
    for k in cfg.keys():
        if k.startswith('learner'):
            if cfg[k].get('use_aggregator', False):
                dst_k = 'aggregator' + k[7:]
                cfg.coordinator.interaction.learner[k] = [k, cfg[dst_k].slave.host, cfg[dst_k].slave.port]
            else:
                dst_k = k
                cfg.coordinator.interaction.learner[k] = [k, cfg[dst_k].host, cfg[dst_k].port]
    return cfg


def set_actor_interaction_for_coordinator(cfg: EasyDict) -> EasyDict:
    cfg.coordinator.interaction.actor = {}
    for k in cfg.keys():
        if k.startswith('actor'):
            cfg.coordinator.interaction.actor[k] = [k, cfg[k].host, cfg[k].port]
    return cfg


def parallel_transform(
        cfg: dict,
        coordinator_host: Optional[str] = None,
        learner_host: Optional[List[str]] = None,
        actor_host: Optional[List[str]] = None
) -> None:
    coordinator_host = default_host if coordinator_host is None else coordinator_host
    actor_host = default_host if actor_host is None else actor_host
    learner_host = default_host if learner_host is None else learner_host
    cfg = EasyDict(cfg)
    cfg = set_host_port(cfg, coordinator_host, learner_host, actor_host)
    cfg = set_learner_interaction_for_coordinator(cfg)
    cfg = set_actor_interaction_for_coordinator(cfg)
    return cfg


def parallel_transform_slurm(
        cfg: dict,
        coordinator_host: Optional[str] = None,
        learner_node: Optional[List[str]] = None,
        actor_node: Optional[List[str]] = None
) -> None:
    cfg = EasyDict(cfg)
    cfg = set_host_port_slurm(cfg, coordinator_host, learner_node, actor_node)
    cfg = set_learner_interaction_for_coordinator(cfg)
    cfg = set_actor_interaction_for_coordinator(cfg)
    pretty_print(cfg)
    return cfg
