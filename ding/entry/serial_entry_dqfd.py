# from ding.policy.base_policy import Policy
import logging
import os
from copy import deepcopy
from functools import partial
from typing import Union, Optional, List, Any, Tuple

import numpy as np
import torch
from tensorboardX import SummaryWriter

from ding.config import read_config, compile_config
from ding.envs import get_vec_env_setting, create_env_manager
from ding.policy import create_policy, PolicyFactory
from ding.utils import set_pkg_seed
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from dizoo.classic_control.cartpole.config.cartpole_dqfd_config import main_config, create_config  # for testing

# from dizoo.classic_control.cartpole.config.cartpole_dqn_config import main_config_1, create_config_1   # for testing


def serial_pipeline_dqfd(
        input_cfg: Union[str, Tuple[dict, dict]],
        expert_input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        expert_model: Optional[torch.nn.Module] = None,
        max_iterations: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline dqfd entry: we create this serial pipeline in order to\
            implement dqfd in DI-engine. For now, we support the following envs\
            Cartpole, Lunarlander, Pong, Spaceinvader. The demonstration\
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
        - max_iterations (:obj:`Optional[torch.nn.Module]`): Learner's max iteration. Pipeline will stop \
            when reaching this iteration.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
        expert_cfg, expert_create_cfg = read_config(expert_input_cfg)
    else:
        cfg, create_cfg = input_cfg
        expert_cfg, expert_create_cfg = expert_input_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    expert_create_cfg.policy.type = expert_create_cfg.policy.type + '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)
    expert_cfg = compile_config(
        expert_cfg, seed=seed, env=env_fn, auto=True, create_cfg=expert_create_cfg, save_cfg=True
    )
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
    # expert_model = DQN(**cfg.policy.model)
    expert_policy = create_policy(expert_cfg.policy, model=expert_model, enable_field=['collect', 'command'])
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    # model = DQN(**cfg.policy.model)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])
    if cfg.policy.cuda and torch.cuda.is_available():  # cpu->gpu 0
        expert_policy.collect_mode.load_state_dict(
            torch.load(expert_cfg.policy.collect.demonstration_info_path, map_location=lambda storage, loc: storage.cuda(0))
        )
    else:
        expert_policy.collect_mode.load_state_dict(
            torch.load(expert_cfg.policy.collect.demonstration_info_path, map_location='cpu')
        )
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
    if expert_cfg.policy.learn.expert_replay_buffer_size != 0:  # for ablation study
        # dummy_variable = deepcopy(cfg.policy.other.replay_buffer)
        # dummy_variable['replay_buffer_size'] = cfg.policy.learn.expert_replay_buffer_size
        # expert_buffer = create_buffer(dummy_variable, tb_logger=tb_logger, exp_name=cfg.exp_name)

        expert_buffer = create_buffer(expert_cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)

        expert_data = expert_collector.collect(
            n_sample=cfg.policy.learn.expert_replay_buffer_size, policy_kwargs=expert_collect_kwargs
        )
        for i in range(len(expert_data)):
            expert_data[i]['is_expert'] = 1  # set is_expert flag(expert 1, agent 0)
        expert_buffer.push(expert_data, cur_collector_envstep=0)
        for _ in range(cfg.policy.learn.per_train_iter_k):
            if evaluator.should_eval(learner.train_iter):
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                if stop:
                    break
            # Learn policy from collected data
            # Expert_learner will train ``update_per_collect == 1`` times in one iteration.
            train_data = expert_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            learner.train(train_data, collector.envstep)
            if learner.policy.get_attribute('priority'):
                expert_buffer.update(learner.priority_info)
        learner.priority_info = {}
    # Accumulate plenty of data at the beginning of training.
    if cfg.policy.get('random_collect_size', 0) > 0:
        action_space = collector_env.env_info().act_space
        random_policy = PolicyFactory.get_random_policy(policy.collect_mode, action_space=action_space)
        collector.reset_policy(random_policy)
        collect_kwargs = commander.step()
        new_data = collector.collect(n_sample=cfg.policy.random_collect_size, policy_kwargs=collect_kwargs)
        for i in range(len(new_data)):
            new_data[i]['is_expert'] = 0  # set is_expert flag(expert 1, agent 0)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        collector.reset_policy(policy.collect_mode)
    for _ in range(max_iterations):
        collect_kwargs = commander.step()
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        # Collect data by default config n_sample/n_episode
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        for i in range(len(new_data)):
            new_data[i]['is_expert'] = 0  # set is_expert flag(expert 1, agent 0)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # Learn policy from collected data
        for i in range(cfg.policy.learn.update_per_collect):
            if cfg.policy.learn.expert_replay_buffer_size != 0:
                # Learner will train ``update_per_collect`` times in one iteration.
                # The hyperparameter pho, the demo ratio, control the propotion of data coming\
                # from expert demonstrations versus from the agent's own experience.
                stats = np.random.choice(
                    (learner.policy.get_attribute('batch_size')), size=(learner.policy.get_attribute('batch_size'))
                ) < (
                    learner.policy.get_attribute('batch_size')
                ) * cfg.policy.collect.pho  # torch.rand((learner.policy.get_attribute('batch_size')))\
                # < cfg.policy.collect.pho
                expert_batch_size = stats[stats].shape[0]
                demo_batch_size = (learner.policy.get_attribute('batch_size')) - expert_batch_size
                train_data = replay_buffer.sample(demo_batch_size, learner.train_iter)
                train_data_demonstration = expert_buffer.sample(expert_batch_size, learner.train_iter)
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
                    # When collector, set replay_buffer_idx and replay_unique_id for each data item, priority = 1.\
                    # When learner, assign priority for each data item according their loss
                    learner.priority_info_agent = deepcopy(learner.priority_info)
                    learner.priority_info_expert = deepcopy(learner.priority_info)
                    learner.priority_info_agent['priority'] = learner.priority_info['priority'][0:demo_batch_size]
                    learner.priority_info_agent['replay_buffer_idx'] = learner.priority_info['replay_buffer_idx'][
                        0:demo_batch_size]
                    learner.priority_info_agent['replay_unique_id'] = learner.priority_info['replay_unique_id'][
                        0:demo_batch_size]
                    learner.priority_info_expert['priority'] = learner.priority_info['priority'][demo_batch_size:]
                    learner.priority_info_expert['replay_buffer_idx'] = learner.priority_info['replay_buffer_idx'][
                        demo_batch_size:]
                    learner.priority_info_expert['replay_unique_id'] = learner.priority_info['replay_unique_id'][
                        demo_batch_size:]
                    # Expert data and demo data update their priority separately.
                    replay_buffer.update(learner.priority_info_agent)
                    expert_buffer.update(learner.priority_info_expert)
            else:
                # Learner will train ``update_per_collect`` times in one iteration.
                train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
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
        if cfg.policy.on_policy:
            # On-policy algorithm must clear the replay buffer.
            replay_buffer.clear()

    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy


if __name__ == '__main__':
    main_config_1 = deepcopy(main_config)
    main_config_2 = deepcopy(create_config)
    serial_pipeline_dqfd([main_config, create_config], [main_config_1, main_config_2], seed=0)
'''
#from ding.policy.base_policy import Policy
from typing import Union, Optional, List, Any, Tuple
import os
import torch
import numpy as np
import logging
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.config import read_config, compile_config
from ding.policy import create_policy, PolicyFactory
from ding.utils import set_pkg_seed
from ding.model import DQN
from copy import deepcopy
from dizoo.classic_control.cartpole.config.cartpole_dqfd_config import main_config, create_config  # for testing


def serial_pipeline_dqfd(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        expert_model: Optional[torch.nn.Module] = None,
        max_iterations: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline dqfd entry: we create this serial pipeline in order to\
            implement dqfd in DI-engine. For now, we support the following envs\
            Cartpole, Lunarlander, Pong, Spaceinvader. The demonstration\
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
        - max_iterations (:obj:`Optional[torch.nn.Module]`): Learner's max iteration. Pipeline will stop \
            when reaching this iteration.
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
    expert_collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    expert_collector_env.seed(cfg.seed)
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    #expert_model = DQN(**cfg.policy.model)
    expert_policy = create_policy(cfg.policy, model=expert_model, enable_field=['collect'])
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    #model = DQN(**cfg.policy.model)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])
    expert_policy.collect_mode.load_state_dict(
        torch.load(cfg.policy.collect.demonstration_info_path, map_location='cpu')
    )
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
    commander = BaseSerialCommander(
            cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode
        )
    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')
    if cfg.policy.learn.expert_replay_buffer_size != 0:  # for ablation study
        dummy_variable = deepcopy(cfg.policy.other.replay_buffer)
        dummy_variable['replay_buffer_size'] = cfg.policy.learn.expert_replay_buffer_size
        expert_buffer = create_buffer(dummy_variable, tb_logger=tb_logger, exp_name=cfg.exp_name)
        expert_data = expert_collector.collect(
            n_sample=cfg.policy.learn.expert_replay_buffer_size, policy_kwargs={'eps': -1}
        )
        for i in range(len(expert_data)):
            expert_data[i]['is_expert'] = 1  # set is_expert flag(expert 1, agent 0)
        expert_buffer.push(expert_data, cur_collector_envstep=0)
        for _ in range(cfg.policy.learn.per_train_iter_k):
            if evaluator.should_eval(learner.train_iter):
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                if stop:
                    break
            # Learn policy from collected data
            # Expert_learner will train ``update_per_collect == 1`` times in one iteration.
            train_data = expert_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            learner.train(train_data, collector.envstep)
            if learner.policy.get_attribute('priority'):
                expert_buffer.update(learner.priority_info)
        learner.priority_info = {}
    # Accumulate plenty of data at the beginning of training.
    if cfg.policy.get('random_collect_size', 0) > 0:
        action_space = collector_env.env_info().act_space
        random_policy = PolicyFactory.get_random_policy(policy.collect_mode, action_space=action_space)
        collector.reset_policy(random_policy)
        collect_kwargs = commander.step()
        new_data = collector.collect(n_sample=cfg.policy.random_collect_size, policy_kwargs=collect_kwargs)
        for i in range(len(new_data)):
            new_data[i]['is_expert'] = 0  # set is_expert flag(expert 1, agent 0)
        replay_buffer.push(new_data, cur_collector_envstep=0)
        collector.reset_policy(policy.collect_mode)
    for _ in range(max_iterations):
        collect_kwargs = commander.step()
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        # Collect data by default config n_sample/n_episode
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        for i in range(len(new_data)):
            new_data[i]['is_expert'] = 0  # set is_expert flag(expert 1, agent 0)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # Learn policy from collected data
        for i in range(cfg.policy.learn.update_per_collect):
            if cfg.policy.learn.expert_replay_buffer_size != 0:
                # Learner will train ``update_per_collect`` times in one iteration.
                # The hyperparameter pho, the demo ratio, control the propotion of data coming\
                # from expert demonstrations versus from the agent's own experience.
                stats = np.random.choice(
                    (learner.policy.get_attribute('batch_size')), size=(learner.policy.get_attribute('batch_size'))
                ) < (
                    learner.policy.get_attribute('batch_size')
                ) * cfg.policy.collect.pho  # torch.rand((learner.policy.get_attribute('batch_size')))\
                # < cfg.policy.collect.pho
                expert_batch_size = stats[stats].shape[0]
                demo_batch_size = (learner.policy.get_attribute('batch_size')) - expert_batch_size
                train_data = replay_buffer.sample(demo_batch_size, learner.train_iter)
                train_data_demonstration = expert_buffer.sample(expert_batch_size, learner.train_iter)
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
                    # When collector, set replay_buffer_idx and replay_unique_id for each data item, priority = 1.\
                    # When learner, assign priority for each data item according their loss
                    learner.priority_info_agent = deepcopy(learner.priority_info)
                    learner.priority_info_expert = deepcopy(learner.priority_info)
                    learner.priority_info_agent['priority'] = learner.priority_info['priority'][0:demo_batch_size]
                    learner.priority_info_agent['replay_buffer_idx'] = learner.priority_info['replay_buffer_idx'][
                        0:demo_batch_size]
                    learner.priority_info_agent['replay_unique_id'] = learner.priority_info['replay_unique_id'][
                        0:demo_batch_size]
                    learner.priority_info_expert['priority'] = learner.priority_info['priority'][demo_batch_size:]
                    learner.priority_info_expert['replay_buffer_idx'] = learner.priority_info['replay_buffer_idx'][
                        demo_batch_size:]
                    learner.priority_info_expert['replay_unique_id'] = learner.priority_info['replay_unique_id'][
                        demo_batch_size:]
                    # Expert data and demo data update their priority separately.
                    replay_buffer.update(learner.priority_info_agent)
                    expert_buffer.update(learner.priority_info_expert)
            else: 
                # Learner will train ``update_per_collect`` times in one iteration.
                train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
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
        if cfg.policy.on_policy:
            # On-policy algorithm must clear the replay buffer.
            replay_buffer.clear()

    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy


if __name__ == '__main__':
    serial_pipeline_dqfd([main_config, create_config], seed=0)
'''
