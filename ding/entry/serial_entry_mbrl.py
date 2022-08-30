from typing import Union, Optional, List, Any, Tuple
import torch
import os
from functools import partial

from tensorboardX import SummaryWriter
from copy import deepcopy

from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    get_buffer_cls, create_serial_collector
from ding.world_model import WorldModel
from ding.worker import IBuffer
from ding.envs import get_vec_env_setting, create_env_manager
from ding.config import read_config, compile_config
from ding.utils import set_pkg_seed, deep_merge_dicts
from ding.policy import create_policy
from ding.world_model import create_world_model
from ding.entry.utils import random_collect


def mbrl_entry_setup(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
) -> Tuple:
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)

    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting

    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    # create logger
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))

    # create world model
    world_model = create_world_model(cfg.world_model, env_fn(cfg.env), tb_logger)

    # create policy
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])

    # create worker
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
    env_buffer = create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, env_buffer, policy.command_mode
    )

    return (
        cfg,
        policy,
        world_model,
        env_buffer,
        learner,
        collector,
        collector_env,
        evaluator,
        commander,
        tb_logger,
    )


def create_img_buffer(
        cfg: dict, input_cfg: Union[str, Tuple[dict, dict]], world_model: WorldModel, tb_logger: 'SummaryWriter'
) -> IBuffer:  # noqa
    if isinstance(input_cfg, str):
        _, create_cfg = read_config(input_cfg)
    else:
        _, create_cfg = input_cfg
    img_buffer_cfg = cfg.world_model.other.imagination_buffer
    img_buffer_cfg.update(create_cfg.imagination_buffer)
    buffer_cls = get_buffer_cls(img_buffer_cfg)
    cfg.world_model.other.imagination_buffer.update(deep_merge_dicts(buffer_cls.default_config(), img_buffer_cfg))
    if img_buffer_cfg.type == 'elastic':
        img_buffer_cfg.set_buffer_size = world_model.buffer_size_scheduler
    img_buffer = create_buffer(cfg.world_model.other.imagination_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
    return img_buffer


def serial_pipeline_dyna(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry for dyna-style model-based RL.
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
    cfg, policy, world_model, env_buffer, learner, collector, collector_env, evaluator, commander, tb_logger = \
        mbrl_entry_setup(input_cfg, seed, env_setting, model)

    img_buffer = create_img_buffer(cfg, input_cfg, world_model, tb_logger)

    learner.call_hook('before_run')

    if cfg.policy.get('random_collect_size', 0) > 0:
        random_collect(cfg.policy, policy, collector, collector_env, commander, env_buffer)

    while True:
        collect_kwargs = commander.step()
        # eval the policy
        if evaluator.should_eval(collector.envstep):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break

        # fill environment buffer
        data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        env_buffer.push(data, cur_collector_envstep=collector.envstep)

        # eval&train world model and fill imagination buffer
        if world_model.should_eval(collector.envstep):
            world_model.eval(env_buffer, collector.envstep, learner.train_iter)
        if world_model.should_train(collector.envstep):
            world_model.train(env_buffer, collector.envstep, learner.train_iter)
            world_model.fill_img_buffer(
                policy.collect_mode, env_buffer, img_buffer, collector.envstep, learner.train_iter
            )

        for i in range(cfg.policy.learn.update_per_collect):
            batch_size = learner.policy.get_attribute('batch_size')
            train_data = world_model.sample(env_buffer, img_buffer, batch_size, learner.train_iter)
            learner.train(train_data, collector.envstep)

        if cfg.policy.on_policy:
            # On-policy algorithm must clear the replay buffer.
            env_buffer.clear()
            img_buffer.clear()

        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    learner.call_hook('after_run')

    return policy


def serial_pipeline_dream(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry for dreamer-style model-based RL.
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
    cfg, policy, world_model, env_buffer, learner, collector, collector_env, evaluator, commander, tb_logger = \
        mbrl_entry_setup(input_cfg, seed, env_setting, model)

    learner.call_hook('before_run')

    if cfg.policy.get('random_collect_size', 0) > 0:
        random_collect(cfg.policy, policy, collector, collector_env, commander, env_buffer)

    while True:
        collect_kwargs = commander.step()
        # eval the policy
        if evaluator.should_eval(collector.envstep):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break

        # fill environment buffer
        data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        env_buffer.push(data, cur_collector_envstep=collector.envstep)

        # eval&train world model and fill imagination buffer
        if world_model.should_eval(collector.envstep):
            world_model.eval(env_buffer, collector.envstep, learner.train_iter)
        if world_model.should_train(collector.envstep):
            world_model.train(env_buffer, collector.envstep, learner.train_iter)

        update_per_collect = cfg.policy.learn.update_per_collect // world_model.rollout_length_scheduler(
            collector.envstep
        )
        update_per_collect = max(1, update_per_collect)
        for i in range(update_per_collect):
            batch_size = learner.policy.get_attribute('batch_size')
            train_data = env_buffer.sample(batch_size, learner.train_iter)
            # dreamer-style: use pure on-policy imagined rollout to train policy,
            # which depends on the current envstep to decide the rollout length
            learner.train(
                train_data, collector.envstep, policy_kwargs=dict(world_model=world_model, envstep=collector.envstep)
            )

        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    learner.call_hook('after_run')

    return policy
