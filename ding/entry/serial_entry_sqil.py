from typing import Union, Optional, List, Any, Tuple
import os
import torch
from ditk import logging
from functools import partial
from tensorboardX import SummaryWriter
from copy import deepcopy

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from .utils import random_collect


def serial_pipeline_sqil(
        input_cfg: Union[str, Tuple[dict, dict]],
        expert_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        expert_model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline sqil entry: we create this serial pipeline in order to\
            implement SQIL in DI-engine. For now, we support the following envs\
            Cartpole, Lunarlander, Pong, Spaceinvader, Qbert. The demonstration\
            data come from the expert model. We use a well-trained model to \
            generate demonstration data online
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - expert_model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.\
            The default model is DQN(**cfg.policy.model)
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
        - max_env_step (:obj:`Optional[int]`): Maximum collected environment interaction steps.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
        expert_cfg, expert_create_cfg = read_config(expert_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
        expert_cfg, expert_create_cfg = expert_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    expert_create_cfg.policy.type = expert_create_cfg.policy.type + '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)
    expert_cfg = compile_config(
        expert_cfg, seed=seed, env=env_fn, auto=True, create_cfg=expert_create_cfg, save_cfg=True
    )
    # expert config must have the same `n_sample`. The line below ensure we do not need to modify the expert configs
    expert_cfg.policy.collect.n_sample = cfg.policy.collect.n_sample
    # Create main components: env, policy
    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    expert_collector_env = create_env_manager(
        expert_cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg]
    )
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    expert_collector_env.seed(cfg.seed)
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    expert_policy = create_policy(expert_cfg.policy, model=expert_model, enable_field=['collect', 'command'])
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])
    expert_policy.collect_mode.load_state_dict(torch.load(cfg.policy.collect.model_path, map_location='cpu'))
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
    expert_collector = create_serial_collector(
        expert_cfg.policy.collect.collector,
        env=expert_collector_env,
        policy=expert_policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=expert_cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
    expert_buffer = create_buffer(expert_cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode
    )
    expert_commander = BaseSerialCommander(
        expert_cfg.policy.other.commander, learner, expert_collector, evaluator, replay_buffer,
        expert_policy.command_mode
    )  # we create this to avoid the issue of eps, this is an issue due to the sample collector part.
    expert_collect_kwargs = expert_commander.step()
    if 'eps' in expert_collect_kwargs:
        expert_collect_kwargs['eps'] = -1
    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')

    # Accumulate plenty of data at the beginning of training.
    if cfg.policy.get('random_collect_size', 0) > 0:
        random_collect(cfg.policy, policy, collector, collector_env, commander, replay_buffer)
    if cfg.policy.get('expert_random_collect_size', 0) > 0:
        random_collect(
            expert_cfg.policy, expert_policy, expert_collector, expert_collector_env, expert_commander, expert_buffer
        )
    while True:
        collect_kwargs = commander.step()
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        # Collect data by default config n_sample/n_episode
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        expert_data = expert_collector.collect(
            train_iter=learner.train_iter, policy_kwargs=expert_collect_kwargs
        )  # policy_kwargs={'eps': -1}
        for i in range(len(new_data)):
            device_1 = new_data[i]['obs'].device
            device_2 = expert_data[i]['obs'].device
            new_data[i]['reward'] = torch.zeros(cfg.policy.nstep).to(device_1)
            expert_data[i]['reward'] = torch.ones(cfg.policy.nstep).to(device_2)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        expert_buffer.push(expert_data, cur_collector_envstep=collector.envstep)
        # Learn policy from collected data
        for i in range(cfg.policy.learn.update_per_collect):
            # Learner will train ``update_per_collect`` times in one iteration.
            train_data = replay_buffer.sample((learner.policy.get_attribute('batch_size')) // 2, learner.train_iter)
            train_data_demonstration = expert_buffer.sample(
                (learner.policy.get_attribute('batch_size')) // 2, learner.train_iter
            )
            if train_data is None:
                # It is possible that replay buffer's data count is too few to train ``update_per_collect`` times
                logging.warning(
                    "Replay buffer's data can only train for {} steps. ".format(i) +
                    "You can modify data collect config, e.g. increasing n_sample, n_episode."
                )
                break
            train_data = train_data + train_data_demonstration
            learner.train(train_data, collector.envstep)
            if learner.policy.get_attribute('priority'):
                replay_buffer.update(learner.priority_info)
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy
