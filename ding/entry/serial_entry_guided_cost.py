from typing import Union, Optional, List, Any, Tuple
import os
import copy
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
from ding.reward_model import create_reward_model
from ding.utils import set_pkg_seed, save_file
from .utils import random_collect


def serial_pipeline_guided_cost(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        expert_model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline guided cost: we create this serial pipeline in order to\
            implement guided cost learning in DI-engine. For now, we support the following envs\
            Cartpole, Lunarlander, Hopper, Halfcheetah, Walker2d. The demonstration\
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
    else:
        cfg, create_cfg = deepcopy(input_cfg)
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)
    # Create main components: env, policy
    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    expert_collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    expert_collector_env.seed(cfg.seed)
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    expert_policy = create_policy(cfg.policy, model=expert_model, enable_field=['learn', 'collect'])
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
        cfg.policy.collect.collector,
        env=expert_collector_env,
        policy=expert_policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
    expert_buffer = create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
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
    dirname = cfg.exp_name + '/reward_model'
    if not os.path.exists(dirname):
        try:
            os.mkdir(dirname)
        except FileExistsError:
            pass
    while True:
        collect_kwargs = commander.step()
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        # Collect data by default config n_sample/n_episode
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        # NOTE: deepcopy data is very important,
        # otherwise the data in the replay buffer will be incorrectly modified.
        # NOTE: this line cannot move to line130, because in line134 the data may be modified in-place.
        train_data = copy.deepcopy(new_data)
        expert_data = expert_collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        expert_buffer.push(expert_data, cur_collector_envstep=expert_collector.envstep)
        # Learn policy from collected data
        for i in range(cfg.reward_model.update_per_collect):
            expert_demo = expert_buffer.sample(cfg.reward_model.batch_size, learner.train_iter)
            samp = replay_buffer.sample(cfg.reward_model.batch_size, learner.train_iter)
            reward_model.train(expert_demo, samp, learner.train_iter, collector.envstep)
        for i in range(cfg.policy.learn.update_per_collect):
            # Learner will train ``update_per_collect`` times in one iteration.
            _ = reward_model.estimate(train_data)
            if train_data is None:
                # It is possible that replay buffer's data count is too few to train ``update_per_collect`` times
                logging.warning(
                    "Replay buffer's data can only train for {} steps. ".format(i) +
                    "You can modify data collect config, e.g. increasing n_sample, n_episode."
                )
                break
            learner.train(train_data, collector.envstep)
            if learner.policy.get_attribute('priority'):
                replay_buffer.update(learner.priority_info)
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break
        # save reward model
        if learner.train_iter % cfg.reward_model.store_model_every_n_train == 0:
            #if learner.train_iter%5000 == 0:
            path = os.path.join(dirname, 'iteration_{}.pth.tar'.format(learner.train_iter))
            state_dict = reward_model.state_dict_reward_model()
            save_file(path, state_dict)
    path = os.path.join(dirname, 'final_model.pth.tar')
    state_dict = reward_model.state_dict_reward_model()
    save_file(path, state_dict)
    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy
