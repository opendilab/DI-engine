from typing import Optional, List, NoReturn
import copy
import json
import os

import yaml
from easydict import EasyDict

from nervex.utils import find_free_port, find_free_port_slurm, node_to_partition, node_to_host, pretty_print
from app_zoo.classic_control.cartpole.config.parallel import cartpole_dqn_config

default_host = '0.0.0.0'
default_port = 22270


def set_host_port(cfg: EasyDict, coordinator_host: str, learner_host: str, collector_host: str) -> EasyDict:
    cfg.coordinator.host = coordinator_host
    if cfg.coordinator.port == 'auto':
        cfg.coordinator.port = find_free_port(coordinator_host)
    learner_count = 0
    collector_count = 0
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
        if k.startswith('collector'):
            if cfg[k].host == 'auto':
                if isinstance(collector_host, list):
                    cfg[k].host = collector_host[collector_count]
                    collector_count += 1
                elif isinstance(collector_host, str):
                    cfg[k].host = collector_host
                else:
                    raise TypeError("not support collector_host type: {}".format(collector_host))
            if cfg[k].port == 'auto':
                cfg[k].port = find_free_port(cfg[k].host)
    return cfg


def set_host_port_slurm(cfg: EasyDict, coordinator_host: str, learner_node: list, collector_node: list) -> EasyDict:
    cfg.coordinator.host = coordinator_host
    if cfg.coordinator.port == 'auto':
        cfg.coordinator.port = find_free_port(coordinator_host)
    if isinstance(learner_node, str):
        learner_node = [learner_node]
    if isinstance(collector_node, str):
        collector_node = [collector_node]
    learner_count, collector_count = 0, 0
    learner_multi = {}
    for k in cfg.keys():
        if learner_node is not None and k.startswith('learner'):
            node = learner_node[learner_count % len(learner_node)]
            cfg[k].node = node
            cfg[k].partition = node_to_partition(node)
            repeat_num = cfg[k].get('repeat_num', 1)
            if cfg[k].host == 'auto':
                cfg[k].host = node_to_host(node)
            if cfg[k].port == 'auto':
                if repeat_num == 1:
                    cfg[k].port = find_free_port_slurm(node)
                    learner_multi[k] = False
                else:
                    cfg[k].port = [find_free_port_slurm(node) for _ in range(repeat_num)]
                    learner_multi[k] = True
            learner_count += 1
        if collector_node is not None and k.startswith('collector'):
            node = collector_node[collector_count % len(collector_node)]
            cfg[k].node = node
            cfg[k].partition = node_to_partition(node)
            if cfg[k].host == 'auto':
                cfg[k].host = node_to_host(node)
            if cfg[k].port == 'auto':
                cfg[k].port = find_free_port_slurm(node)
            collector_count += 1
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


def set_host_port_k8s(cfg: EasyDict) -> EasyDict:
    cfg.coordinator.host = default_host
    cfg.coordinator.port = default_port + 3
    base_learner_cfg = None
    base_collector_cfg = None
    for k in cfg.keys():
        if k.startswith('learner'):
            # create the base learner config
            if base_learner_cfg is None:
                base_learner_cfg = copy.deepcopy(cfg[k])
                base_learner_cfg.host = default_host
                base_learner_cfg.port = default_port + 1
            cfg[k].port = default_port + 1
        elif k.startswith('collector'):
            # create the base collector config
            if base_collector_cfg is None:
                base_collector_cfg = copy.deepcopy(cfg[k])
                base_collector_cfg.host = default_host
                base_collector_cfg.port = default_port + 0
            cfg[k].port = default_port + 0
    cfg['learner'] = base_learner_cfg
    cfg['collector'] = base_collector_cfg
    return cfg


def set_learner_interaction_for_coordinator(cfg: EasyDict) -> EasyDict:
    cfg.coordinator.learner = {}
    for k in cfg.keys():
        if k.startswith('learner'):
            if cfg[k].get('use_aggregator', False):
                dst_k = 'aggregator' + k[7:]
                cfg.coordinator.learner[k] = [k, cfg[dst_k].slave.host, cfg[dst_k].slave.port]
            else:
                dst_k = k
                cfg.coordinator.learner[k] = [k, cfg[dst_k].host, cfg[dst_k].port]
    return cfg


def set_collector_interaction_for_coordinator(cfg: EasyDict) -> EasyDict:
    cfg.coordinator.collector = {}
    for k in cfg.keys():
        if k.startswith('collector'):
            cfg.coordinator.collector[k] = [k, cfg[k].host, cfg[k].port]
    return cfg


def set_system_cfg(cfg: EasyDict) -> EasyDict:
    learner_num = cfg.main.policy.learn.learner.learner_num
    collector_num = cfg.main.policy.collect.collector.collector_num
    path_data = cfg.system.path_data
    path_policy = cfg.system.path_policy
    coordinator_cfg = cfg.system.coordinator
    communication_mode = cfg.system.communication_mode
    assert communication_mode in ['auto'], communication_mode
    learner_multi_gpu = cfg.system.learner_multi_gpu
    new_cfg = dict(coordinator=dict(
        host='auto',
        port='auto',
    ))
    new_cfg['coordinator'].update(coordinator_cfg)
    for i in range(learner_num):
        new_cfg[f'learner{i}'] = dict(
            type=cfg.system.comm_learner.type,
            import_names=cfg.system.comm_learner.import_names,
            host='auto',
            port='auto',
            path_data=path_data,
            path_policy=path_policy,
            multi_gpu=learner_multi_gpu,
        )
    for i in range(collector_num):
        new_cfg[f'collector{i}'] = dict(
            type=cfg.system.comm_collector.type,
            import_names=cfg.system.comm_collector.import_names,
            host='auto',
            port='auto',
            path_data=path_data,
            path_policy=path_policy,
        )
    return EasyDict(new_cfg)


def parallel_transform(
        cfg: dict,
        coordinator_host: Optional[str] = None,
        learner_host: Optional[List[str]] = None,
        collector_host: Optional[List[str]] = None
) -> None:
    coordinator_host = default_host if coordinator_host is None else coordinator_host
    collector_host = default_host if collector_host is None else collector_host
    learner_host = default_host if learner_host is None else learner_host
    cfg = EasyDict(cfg)
    cfg.system = set_system_cfg(cfg)
    cfg.system = set_host_port(cfg.system, coordinator_host, learner_host, collector_host)
    cfg.system = set_learner_interaction_for_coordinator(cfg.system)
    cfg.system = set_collector_interaction_for_coordinator(cfg.system)
    return cfg


def parallel_transform_slurm(
        cfg: dict,
        coordinator_host: Optional[str] = None,
        learner_node: Optional[List[str]] = None,
        collector_node: Optional[List[str]] = None
) -> None:
    cfg = EasyDict(cfg)
    cfg.system = set_system_cfg(cfg)
    cfg.system = set_host_port_slurm(cfg.system, coordinator_host, learner_node, collector_node)
    cfg.system = set_learner_interaction_for_coordinator(cfg.system)
    cfg.system = set_collector_interaction_for_coordinator(cfg.system)
    pretty_print(cfg)
    return cfg


def parallel_transform_k8s(cfg: dict) -> None:
    cfg = EasyDict(cfg)
    cfg.system = set_system_cfg(cfg)
    cfg.system = set_host_port_k8s(cfg.system)
    # learner/collector is created by opereator, so the following field is placeholder
    cfg.system.coordinator.collector = {}
    cfg.system.coordinator.learner = {}
    pretty_print(cfg)
    return cfg


parallel_test_main_config = cartpole_dqn_config
parallel_test_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['app_zoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn_command'),
    comm_learner=dict(
        type='flask_fs',
        import_names=['nervex.worker.learner.comm.flask_fs_learner'],
    ),
    comm_collector=dict(
        type='flask_fs',
        import_names=['nervex.worker.collector.comm.flask_fs_collector'],
    ),
    learner=dict(
        type='base',
        import_names=['nervex.worker.learner.base_learner'],
    ),
    collector=dict(
        type='zergling',
        import_names=['nervex.worker.collector.zergling_collector'],
    ),
    commander=dict(
        type='naive',
        import_names=['nervex.worker.coordinator.base_parallel_commander'],
    ),
)
parallel_test_create_config = EasyDict(parallel_test_create_config)
parallel_test_system_config = dict(
    path_data='.',
    path_policy='.',
    communication_mode='auto',
    learner_multi_gpu=False,
)
parallel_test_system_config = EasyDict(parallel_test_system_config)
