from typing import Union, Optional, List, Any, Callable
import os
import torch
import logging
from functools import partial
from tensorboardX import SummaryWriter

from nervex.envs import get_vec_env_setting, create_env_manager
from nervex.worker import BaseLearner, BaseSerialCollector, BaseSerialEvaluator, BaseSerialCommander
from nervex.config import read_config
from nervex.data import BufferManager
from nervex.policy import create_policy
from .utils import set_pkg_seed


def serial_pipeline(
        cfg: Union[str, dict],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
) -> 'BasePolicy':  # noqa
    r"""
    Overview:
        Serial pipeline entry.
    Arguments:
        - cfg (:obj:`Union[str, dict]`): Config in dict type. ``str`` type means config file path.
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): Subclass of ``BaseEnv``, and config dict.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
    """
    if isinstance(cfg, str):
        cfg = read_config(cfg)
    cfg.policy.policy_type = cfg.policy.policy_type + '_command'
    # Prepare vectorize env
    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env.env_kwargs)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    # Random seed
    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.use_cuda)
    # Create components: policy, learner, collector, evaluator, replay buffer, commander.
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])
    tb_logger = SummaryWriter(os.path.join('./log/', 'serial'))
    learner = BaseLearner(cfg.learner, policy.learn_mode, tb_logger)
    collector = BaseSerialCollector(cfg.collector, collector_env, policy.collect_mode, tb_logger)
    evaluator = BaseSerialEvaluator(cfg.evaluator, evaluator_env, policy.eval_mode, tb_logger)
    replay_buffer = BufferManager(cfg.replay_buffer, tb_logger)
    commander = BaseSerialCommander(
        cfg.get('commander', {}), learner, collector, evaluator, replay_buffer, policy.command_mode
    )
    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')

    # Accumulate plenty of data at the beginning of training.
    if replay_buffer.replay_start_size() > 0:
        collect_kwargs = commander.step()
        new_data = collector.collect_data(
            learner.train_iter, n_sample=replay_buffer.replay_start_size(), policy_kwargs=collect_kwargs
        )
        replay_buffer.push(new_data, cur_collector_envstep=0)
    while True:
        collect_kwargs = commander.step()
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        # Collect data by default config n_sample/n_episode
        new_data = collector.collect_data(learner.train_iter, policy_kwargs=collect_kwargs)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # Learn policy from collected data
        for i in range(cfg.policy.learn.train_iteration):
            # Learner will train ``train_iteration`` times in one iteration.
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is None:
                # It is possible that replay buffer's data count is too few to train ``train_iteration`` times
                logging.warning(
                    "Replay buffer's data can only train for {} steps. ".format(i) +
                    "You can modify data collect config, e.g. increasing n_sample, n_episode."
                )
                break
            learner.train(train_data, collector.envstep)
            if cfg.policy.get('use_priority', False):
                replay_buffer.update(learner.priority_info)
        if cfg.policy.on_policy:
            # On-policy algorithm must clear the replay buffer.
            replay_buffer.clear()

    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy
