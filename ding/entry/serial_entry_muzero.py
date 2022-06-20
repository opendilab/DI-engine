from typing import Union, Optional, List, Any, Tuple
import os
import torch
import logging
from functools import partial
from tensorboardX import SummaryWriter
import numpy as np
from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.worker.collector.base_serial_evaluator_muzero import BaseSerialEvaluatorMuZero as BaseSerialEvaluator

from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from .utils import random_collect
from ding.data.buffer.game_buffer import GameBuffer
from easydict import EasyDict


def serial_pipeline_muzero(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
        game_config: Optional[dict] = None,
) -> 'Policy':  # noqa
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
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)
    # Create main components: env, policy
    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    # replay_buffer = create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
    # gamebuffer_config = EasyDict(
    #     dict(
    #         batch_size=10,
    #         transition_num=20,
    #         priority_prob_alpha=0.5,
    #         total_transitions=10000,
    #         num_unroll_steps=5,
    #         td_steps=5,
    #         auto_td_steps=int(0.3*2e5),
    #         stacked_observations=4,
    #         device='cpu',
    #         use_root_value=True,
    #         mini_infer_size = 2,
    #
    #     )
    # )
    # gamebuffer_config=game_config
    replay_buffer = GameBuffer(game_config)

    collector = create_serial_collector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        replay_buffer=replay_buffer,
        game_config=game_config
    )
    evaluator = BaseSerialEvaluator(
        cfg.policy.eval.evaluator,
        evaluator_env,
        policy.eval_mode,
        tb_logger,
        exp_name=cfg.exp_name,
        game_config=game_config
    )

    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode
    )
    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')

    # Accumulate plenty of data at the beginning of training.
    # if cfg.policy.get('random_collect_size', 0) > 0:
    #     random_collect(cfg.policy, policy, collector, collector_env, commander, replay_buffer)
    while True:
        collect_kwargs = commander.step()
        # set temperature for distributions
        collect_kwargs['temperature'] = np.array(
            [game_config.visit_softmax_temperature_fn(trained_steps=learner.train_iter) for _ in range(game_config.env_num)])
        # Evaluate policy performance
        # if evaluator.should_eval(learner.train_iter):
        stop, reward = evaluator.eval(
            learner.save_checkpoint, learner.train_iter, collector.envstep, config=game_config
        )
        if stop:
            break

        # Collect data by default config n_sample/n_episode
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        # replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # Learn policy from collected data
        for i in range(cfg.policy.learn.update_per_collect):
            # Learner will train ``update_per_collect`` times in one iteration.
            train_data = replay_buffer.sample_train_data(learner.policy.get_attribute('batch_size'), policy)
            if train_data is None:
                # It is possible that replay buffer's data count is too few to train ``update_per_collect`` times
                logging.warning(
                    "Replay buffer's data can only train for {} steps. ".format(i) +
                    "You can modify data collect config, e.g. increasing n_sample, n_episode."
                )
                break
            learner.train(train_data, collector.envstep)
            # if learner.policy.get_attribute('priority'):
            #     replay_buffer.update(learner.priority_info)
            #     replay_buffer.batch_update(indices=learner.priority_info['priority']['indices'], metas={'make_time': \
            #     learner.priority_info['priority']['make_time'], 'batch_priorities': learner.priority_info['priority']['value_priority']})
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy
