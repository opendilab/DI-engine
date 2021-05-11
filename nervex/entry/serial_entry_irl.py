from typing import Union, Optional, List, Any, Callable, Tuple
import os
import torch
import logging
from functools import partial
from tensorboardX import SummaryWriter

from nervex.envs import get_vec_env_setting, create_env_manager
from nervex.worker import BaseLearner, BaseSerialCollector, BaseSerialEvaluator, BaseSerialCommander
from nervex.config import read_config, compile_config
from nervex.data import BufferManager
from nervex.policy import create_policy
from nervex.irl_utils import create_irl_model
from nervex.utils import set_pkg_seed


def serial_pipeline_irl(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
) -> 'BasePolicy':  # noqa
    r"""
    Overview:
        Serial pipeline entry.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    # TODO(nyz) when env_setting is not None
    assert env_setting is None  # temporally
    create_cfg.policy.type += '_command'
    cfg = compile_config(cfg, auto=True, create_cfg=create_cfg)

    # Create main components: env, policy
    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(seed)
    evaluator_env.seed(seed)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])

    # Create components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./log/', 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger)
    collector = BaseSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger)
    evaluator = BaseSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger)
    replay_buffer = BufferManager(cfg.policy.other.replay_buffer, tb_logger)
    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode
    )
    reward_model = create_irl_model(cfg.irl, policy.collect_mode.get_attribute('device'), tb_logger)

    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')
    # Accumulate plenty of data at the beginning of training.
    if replay_buffer.replay_buffer_start_size() > 0:
        collect_kwargs = commander.step()
        new_data = collector.collect_data(
            learner.train_iter, n_sample=replay_buffer.replay_buffer_start_size(), policy_kwargs=collect_kwargs
        )
        replay_buffer.push(new_data, cur_collector_envstep=0)
    while True:
        collect_kwargs = commander.step()
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        new_data_count, target_new_data_count = 0, cfg.irl.get('target_new_data_count', 1)
        while new_data_count < target_new_data_count:
            new_data = collector.collect_data(learner.train_iter, policy_kwargs=collect_kwargs)
            new_data_count += len(new_data)
            # collect data for reward_model training
            reward_model.collect_data(new_data)
            replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # update reward_model
        reward_model.train()
        reward_model.clear_data()
        # Learn policy from collected data
        for i in range(cfg.policy.learn.update_per_collect):
            # Learner will train ``update_per_collect`` times in one iteration.
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is None:
                # It is possible that replay buffer's data count is too few to train ``update_per_collect`` times
                logging.warning(
                    "Replay buffer's data can only train for {} steps. ".format(i) +
                    "You can modify data collect config, e.g. increasing n_sample, n_episode."
                )
                break
            # update train_data reward
            reward_model.estimate(train_data)
            learner.train(train_data, collector.envstep)
            if cfg.policy.get('priority', False):
                replay_buffer.update(learner.priority_info)
        if cfg.policy.on_policy:
            # On-policy algorithm must clear the replay buffer.
            replay_buffer.clear()
    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy
