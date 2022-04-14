from typing import Union, Optional, List, Any, Tuple
import os
import copy
import torch
import logging
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import get_vec_env_setting, create_env_manager, create_model_env
from ding.model import create_model
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed, read_file, save_file
from .utils import random_collect


def save_ckpt_fn(learner, env_model, envstep):

    dirname = './{}/ckpt'.format(learner.exp_name)
    if not os.path.exists(dirname):
        try:
            os.mkdir(dirname)
        except FileExistsError:
            pass
    policy_prefix = 'policy_envstep_{}_'.format(envstep)
    model_prefix = 'model_envstep_{}_'.format(envstep)

    def model_save_ckpt_fn(ckpt_name):
        """
        Overview:
            Save checkpoint in corresponding path.
            Checkpoint info includes policy state_dict and iter num.
        Arguments:
            - engine (:obj:`BaseLearner`): the BaseLearner which needs to save checkpoint
        """
        path = os.path.join(dirname, ckpt_name)
        state_dict = env_model.state_dict()
        save_file(path, state_dict)
        learner.info('env model save ckpt in {}'.format(path))

    def model_policy_save_ckpt_fn(ckpt_name):
        model_save_ckpt_fn(model_prefix + ckpt_name)
        learner.save_checkpoint(policy_prefix + ckpt_name)

    return model_policy_save_ckpt_fn


def serial_pipeline_mbrl(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry for model-based RL.
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
    # Compile config
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    model_based_cfg = cfg.pop('model_based')
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)

    # Create logger
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))

    # Create env
    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)

    # Create env model
    model_based_cfg.env_model.tb_logger = tb_logger
    env_model = create_model(model_based_cfg.env_model)

    # Create model-based env
    model_env = create_model_env(model_based_cfg.model_env)

    # Create policy
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
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
    imagine_buffer = create_buffer(model_based_cfg.imagine_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode
    )

    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')

    # Accumulate plenty of data before the beginning of training.
    if cfg.policy.get('random_collect_size', 0) > 0:
        random_collect(cfg.policy, policy, collector, collector_env, commander, replay_buffer)
    # Train
    batch_size = learner.policy.get_attribute('batch_size')
    real_ratio = model_based_cfg['real_ratio']
    replay_batch_size = int(batch_size * real_ratio)
    imagine_batch_size = batch_size - replay_batch_size
    eval_buffer = []
    while True:
        collect_kwargs = commander.step()
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(
                save_ckpt_fn(learner, env_model, collector.envstep), learner.train_iter, collector.envstep
            )
            if stop:
                break
        # Collect data by default config n_sample/n_episode
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        eval_buffer.extend(new_data)
        # Eval env_model
        if env_model.should_eval(collector.envstep):
            env_model.eval(eval_buffer, collector.envstep)
            eval_buffer = []
        # Train env_model and use model_env to rollout
        if env_model.should_train(collector.envstep):
            env_model.train(replay_buffer, learner.train_iter, collector.envstep)
            imagine_buffer.update(collector.envstep)
            model_env.rollout(
                env_model, policy.collect_mode, replay_buffer, imagine_buffer, collector.envstep, learner.train_iter
            )
            policy._rollout_length = model_env._set_rollout_length(collector.envstep)
        # Learn policy from collected data
        for i in range(cfg.policy.learn.update_per_collect):
            # Learner will train ``update_per_collect`` times in one iteration.
            replay_train_data = replay_buffer.sample(replay_batch_size, learner.train_iter)
            imagine_batch_data = imagine_buffer.sample(imagine_batch_size, learner.train_iter)
            if replay_train_data is None or imagine_batch_data is None:
                break
            train_data = replay_train_data + imagine_batch_data
            learner.train(train_data, collector.envstep)
            # Priority is not support
            # if learner.policy.get_attribute('priority'):
            #     replay_buffer.update(learner.priority_info)
        if cfg.policy.on_policy:
            # On-policy algorithm must clear the replay buffer.
            replay_buffer.clear()
            imagine_buffer.clear()
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy
