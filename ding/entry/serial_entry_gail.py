from typing import Optional, Tuple
import os
import torch
from ditk import logging
from functools import partial
from tensorboardX import SummaryWriter
from copy import deepcopy
import numpy as np

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.reward_model import create_reward_model
from ding.utils import set_pkg_seed
from ding.entry import collect_demo_data
from ding.utils import save_file
from .utils import random_collect


def save_reward_model(path, reward_model, weights_name='best'):
    path = os.path.join(path, 'reward_model', 'ckpt')
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
    path = os.path.join(path, 'ckpt_{}.pth.tar'.format(weights_name))
    state_dict = reward_model.state_dict()
    save_file(path, state_dict)
    print('Saved reward model ckpt in {}'.format(path))


def serial_pipeline_gail(
        input_cfg: Tuple[dict, dict],
        expert_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
        collect_data: bool = True,
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry for GAIL reward model.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - expert_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Expert config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
        - max_env_step (:obj:`Optional[int]`): Maximum collected environment interaction steps.
        - collect_data (:obj:`bool`): Collect expert data.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
    if isinstance(expert_cfg, str):
        expert_cfg, expert_create_cfg = read_config(expert_cfg)
    else:
        expert_cfg, expert_create_cfg = expert_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg, save_cfg=True)
    if 'data_path' not in cfg.reward_model:
        cfg.reward_model.data_path = cfg.exp_name
    # Load expert data
    if collect_data:
        if expert_cfg.policy.get('other', None) is not None and expert_cfg.policy.other.get('eps', None) is not None:
            expert_cfg.policy.other.eps.collect = -1
        if expert_cfg.policy.get('load_path', None) is None:
            expert_cfg.policy.load_path = cfg.reward_model.expert_model_path
        collect_demo_data(
            (expert_cfg, expert_create_cfg),
            seed,
            state_dict_path=expert_cfg.policy.load_path,
            expert_data_path=cfg.reward_model.data_path + '/expert_data.pkl',
            collect_count=cfg.reward_model.collect_count
        )
    # Create main components: env, policy
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = create_serial_collector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode
    )
    reward_model = create_reward_model(cfg.reward_model, policy.collect_mode.get_attribute('device'), tb_logger)

    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')

    # Accumulate plenty of data at the beginning of training.
    if cfg.policy.get('random_collect_size', 0) > 0:
        random_collect(cfg.policy, policy, collector, collector_env, commander, replay_buffer)
    best_reward = -np.inf
    while True:
        collect_kwargs = commander.step()
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            reward_mean = np.array([r['final_eval_reward'] for r in reward]).mean()
            if reward_mean >= best_reward:
                save_reward_model(cfg.exp_name, reward_model, 'best')
                best_reward = reward_mean
            if stop:
                break
        new_data_count, target_new_data_count = 0, cfg.reward_model.get('target_new_data_count', 1)
        while new_data_count < target_new_data_count:
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
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
            # update train_data reward using the augmented reward
            train_data_augmented = reward_model.estimate(train_data)
            learner.train(train_data_augmented, collector.envstep)
            if learner.policy.get_attribute('priority'):
                replay_buffer.update(learner.priority_info)
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
    save_reward_model(cfg.exp_name, reward_model, 'last')
    # evaluate
    # evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
    return policy
