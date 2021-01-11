from typing import Optional, List
from easydict import EasyDict
from nervex.utils import find_free_port

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


def set_learner_aggregator_config(cfg: EasyDict) -> EasyDict:
    learner_names = []
    for k in cfg.keys():
        if k.startswith('learner'):
            learner_names.append(k[7:])
    aggregator_cfgs = {}
    for n in learner_names:
        if isinstance(list, cfg['learner' + n].port):
            host = 'auto'
            learner_host = cfg['learner' + n].host
            learner_port = cfg['learner' + n].port
            assert len(learner_host) == len(learner_port)
            learner_interaction_cfg = {i: [i, h, p] for i, (h, p) in enumerate(zip(learner_host, learner_port))}
            aggregator_cfg = dict(
                master=dict(
                    host=host,
                    port=find_free_port(),
                ),
                slave=dict(
                    host=host,
                    port=find_free_port(),
                ),
                learner=learner_interaction_cfg,
            )
            aggregator_cfgs[n] = aggregator_cfg

    for n, c in aggregator_cfgs.items():
        cfg['aggregator' + n] = c
    return cfg


def set_learner_interaction_for_coordinator(cfg: EasyDict) -> EasyDict:
    use_aggregator = cfg.get('use_aggregator', False)
    keyword = 'aggregator' if use_aggregator else 'learner'
    cfg.coordinator.interaction.learner = {}
    for k in cfg.keys():
        if k.startswith(keyword):
            n = k[len(keyword):]
            cfg.coordinator.interaction.learner['learner' + n] = ['learner' + n, cfg[k].host, cfg[k].port]
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
