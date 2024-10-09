import os
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from ditk import logging
from numpy.lib.format import open_memmap
from tensorboardX import SummaryWriter

from qtransformer.algorithm.dataset_qtransformer import ReplayMemoryDataset, SampleData
from ding.config import compile_config, read_config
from ding.envs import (
    AsyncSubprocessEnvManager,
    BaseEnvManager,
    SyncSubprocessEnvManager,
    create_env_manager,
    get_vec_env_setting,
)
from ding.policy import create_policy
from ding.utils import get_rank, set_pkg_seed
from ding.worker import (
    BaseLearner,
    BaseSerialCommander,
    EpisodeSerialCollector,
    InteractionSerialEvaluator,
    create_buffer,
    create_serial_collector,
    create_serial_evaluator,
)


def serial_pipeline_episode(
    input_cfg: Union[str, Tuple[dict, dict]],
    seed: int = 0,
    env_setting: Optional[List[Any]] = None,
    model: Optional[torch.nn.Module] = None,
    max_train_iter: Optional[int] = int(1e10),
    max_env_step: Optional[int] = int(1e10),
    dynamic_seed: Optional[bool] = True,
) -> "Policy":  # noqa
    """
    Overview:
        Serial pipeline entry for off-policy RL.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
        - max_env_step (:obj:`Optional[int]`): Maximum collected environment interaction steps.
        - dynamic_seed(:obj:`Optional[bool]`): set dynamic seed for collector.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
    create_cfg.policy.type = create_cfg.policy.type + "_command"
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(
        cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True
    )
    # Create main components: env, policy
    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    collector_env = create_env_manager(
        cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg]
    )
    evaluator_env = create_env_manager(
        cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg]
    )
    collector_env.seed(cfg.seed, dynamic_seed=dynamic_seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    policy = create_policy(
        cfg.policy, model=model, enable_field=["learn", "collect", "eval", "command"]
    )

    ckpt_path = "/root/code/DI-engine/qtransformer/model/ckpt_best.pth.tar"
    checkpoint = torch.load(ckpt_path)
    policy._model.load_state_dict(checkpoint["model"])

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = (
        SummaryWriter(os.path.join("./{}/log/".format(cfg.exp_name), "serial"))
        if get_rank() == 0
        else None
    )
    learner = BaseLearner(
        cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name
    )
    # collector = create_serial_collector(
    #     cfg.policy.collect.collector,
    #     env=collector_env,
    #     policy=policy.collect_mode,
    #     tb_logger=tb_logger,
    #     exp_name=cfg.exp_name,
    # )

    # collector = EpisodeSerialCollector(
    #     EpisodeSerialCollector.default_config(),
    #     env=evaluator_env,
    #     policy=policy.collect_mode,
    # )
    # evaluator = create_serial_evaluator(
    #     cfg.policy.eval.evaluator,
    #     env=evaluator_env,
    #     policy=policy.eval_mode,
    #     tb_logger=tb_logger,
    #     exp_name=cfg.exp_name,
    # )
    replay_buffer = create_buffer(
        cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name
    )
    commander = BaseSerialCommander(
        cfg.policy.other.commander,
        learner,
        collector,
        None,
        replay_buffer,
        policy.command_mode,
    )
    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook("before_run")

    # Accumulate plenty of data at the beginning of training.
    # if cfg.policy.get("random_collect_size", 0) > 0:
    #     random_collect(
    #         cfg.policy, policy, collector, collector_env, commander, replay_buffer
    #     )
    n_episode = 50
    collected_episode = collector.collect(
        n_episode=n_episode,
        train_iter=collector._collect_print_freq,
        policy_kwargs={"eps": 0.5},
    )
    torch.save(
        collected_episode, "/root/code/DI-engine/qtransformer/model/torchdict_tmp"
    )
    value_test = SampleData(
        memories_dataset_folder="/root/code/DI-engine/qtransformer/model",
        num_episodes=n_episode,
    )
    value_test.transformer("/root/code/DI-engine/qtransformer/model/torchdict_tmp")
