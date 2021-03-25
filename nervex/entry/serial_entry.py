import sys
import copy
import time
from typing import Union, Optional, List, Any
import numpy as np
import torch
import math
import logging

from nervex.worker import BaseLearner, BaseSerialActor, BaseSerialEvaluator, BaseSerialCommander
from nervex.worker import BaseEnvManager, SubprocessEnvManager
from nervex.config import read_config
from nervex.data import BufferManager
from nervex.policy import create_policy
from nervex.envs import get_vec_env_setting
from .utils import set_pkg_seed


def serial_pipeline(
        cfg: Union[str, dict],
        seed: int,
        env_setting: Optional[Any] = None,
        policy_type: Optional[type] = None,
        model: Optional[Union[type, torch.nn.Module]] = None,
        enable_total_log: Optional[bool] = True,
) -> 'BasePolicy':  # noqa
    r"""
    Overview:
        Serial pipeline entry.
    Arguments:
        - cfg (:obj:`Union[str, dict]`): Config in dict type. ``str`` type means config file path.
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[Any]`): Subclass of ``BaseEnv``, and config dict.
        - policy_type (:obj:`Optional[type]`): Subclass of ``Policy``.
        - model (:obj:`Optional[Union[type, torch.nn.Module]]`): Instance or subclass of torch.nn.Module.
        - enable_total_log (:obj:`Optional[bool]`): whether enable total nervex system log
    """
    # Disable some parts nervex system log
    if not enable_total_log:
        actor_log = logging.getLogger('actor_logger')
        actor_log.disabled = True
    if isinstance(cfg, str):
        cfg = read_config(cfg)
    # Default case: Create env_num envs with copies of env cfg.
    # If you want to indicate different cfg for different env, please refer to ``get_vec_env_setting``.
    # Usually, user-defined env must be registered in nervex so that it can be created with config string;
    # Or you can also directly pass in env_fn argument, in some dynamic env class cases.
    manager_cfg = cfg.env.get('manager', {})
    if env_setting is None:
        env_fn, actor_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, actor_env_cfg, evaluator_env_cfg = env_setting
    env_manager_type = BaseEnvManager if cfg.env.env_manager_type == 'base' else SubprocessEnvManager
    actor_env = env_manager_type(
        env_fn=env_fn, env_cfg=actor_env_cfg, env_num=len(actor_env_cfg), manager_cfg=manager_cfg
    )
    evaluator_env = env_manager_type(
        env_fn, env_cfg=evaluator_env_cfg, env_num=len(evaluator_env_cfg), manager_cfg=manager_cfg
    )
    # Random seed
    actor_env.seed(seed)
    evaluator_env.seed(seed)
    set_pkg_seed(seed, use_cuda=cfg.policy.use_cuda)
    # Create components: policy, learner, actor, evaluator, replay buffer, commander.
    if policy_type is None:
        policy_fn = create_policy
    else:
        assert callable(policy_type)
        policy_fn = policy_type
    policy = policy_fn(cfg.policy, model=model)
    learner = BaseLearner(cfg.learner)
    # Append the load path into config
    cfg.learner.load_path = learner.name
    actor = BaseSerialActor(cfg.actor)
    evaluator = BaseSerialEvaluator(cfg.evaluator)
    replay_buffer = BufferManager(cfg.replay_buffer)
    commander = BaseSerialCommander(cfg.commander, learner, actor, evaluator, replay_buffer)
    # Set corresponding env and policy mode.
    actor.env = actor_env
    evaluator.env = evaluator_env
    learner.policy = policy.learn_mode
    actor.policy = policy.collect_mode
    evaluator.policy = policy.eval_mode
    commander.policy = policy.command_mode
    # ==========
    # Main loop
    # ==========
    replay_buffer.start()
    # Max evaluation reward from beginning till now.
    max_eval_reward = float("-inf")
    # Evaluate interval. Will be set to 0 after one evaluation.
    eval_interval = cfg.evaluator.eval_freq
    # How many steps to train in actor's one collection.
    learner_train_step = cfg.policy.learn.train_step
    # Here we assume serial entry and most policy in serial mode mainly focuses on agent buffer.
    # ``enough_data_count``` is just a lower bound estimation. It is possible that replay buffer's data count is
    # greater than this value, but still does not have enough data to train ``train_step`` times.
    enough_data_count = []
    for buffer_name in cfg.replay_buffer.buffer_name:
        enough_data_count.append(
            cfg.policy.learn.batch_size * max(
                cfg.replay_buffer[buffer_name].min_sample_ratio,
                math.ceil(cfg.policy.learn.train_step / cfg.replay_buffer[buffer_name].max_reuse)
            )
        )
    enough_data_count = max(enough_data_count)
    # Accumulate plenty of data at the beginning of training.
    # If "init_data_count" does not exist in config, ``init_data_count`` will be set to ``enough_data_count``.
    init_data_count = cfg.policy.learn.get('init_data_count', enough_data_count)
    # Whether to switch on priority experience replay.
    use_priority = cfg.policy.get('use_priority', False)
    # Learner's before_run hook.
    learner.call_hook('before_run')
    while True:
        commander.step()
        # Evaluate at the beginning of training.
        if eval_interval >= cfg.evaluator.eval_freq:
            stop_flag, eval_reward = evaluator.eval(learner.train_iter)
            eval_interval = 0
            if stop_flag and learner.train_iter > 0:
                # Evaluator's mean episode reward reaches the expected ``stop_val``.
                print(
                    "[nerveX serial pipeline] Your RL agent is converged, you can refer to " +
                    "'log/evaluator/evaluator_logger.txt' for details"
                )
                break
            else:
                if eval_reward > max_eval_reward:
                    learner.save_checkpoint()
                    max_eval_reward = eval_reward
        while True:
            # Actor keeps generating data until replay buffer has enough to sample one batch.
            new_data = actor.generate_data(learner.train_iter)
            replay_buffer.push(new_data)
            target_count = init_data_count if learner.train_iter == 0 else enough_data_count
            if replay_buffer.count() >= target_count:
                break
        for i in range(learner_train_step):
            # Learner will train ``train_step`` times in one iteration.
            # But if replay buffer does not have enough data, program will break and warn.
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is None:
                # As noted above: It is possible that replay buffer's data count is
                # greater than ``target_count```, but still has no enough data to train ``train_step`` times.
                logging.warning(
                    "Replay buffer's data can only train for {} steps. ".format(i) +
                    "You can modify data collect config, e.g. increasing n_sample, n_episode or min_sample_ratio."
                )
                break
            learner.train(train_data)
            eval_interval += 1
            if use_priority:
                replay_buffer.update(learner.priority_info)
        if cfg.policy.on_policy:
            # On-policy algorithm must clear the replay buffer.
            replay_buffer.clear()
    # Learner's after_run hook.
    learner.call_hook('after_run')
    # Close all resources.
    replay_buffer.close()
    learner.close()
    actor.close()
    evaluator.close()
    return policy
