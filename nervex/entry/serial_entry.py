import sys
import copy
import time
from typing import Union, Optional, List, Any
import numpy as np
import torch
import math
import warnings

from nervex.worker import BaseLearner, BaseSerialActor, BaseSerialEvaluator, BaseSerialCommand
from nervex.worker import BaseEnvManager, SubprocessEnvManager
from nervex.utils import read_config
from nervex.data import BufferManager
from nervex.policy import create_policy
from nervex.envs import get_vec_env_setting


def serial_pipeline(
        cfg: Union[str, dict],
        seed: int,
        env_setting: Optional[Any] = None,  # subclass of BaseEnv, and config dict
        policy_type: Optional[type] = None,  # subclass of Policy
        model: Optional[Union[type, torch.nn.Module]] = None,  # instance or subclass of torch.nn.Module
) -> None:
    if isinstance(cfg, str):
        cfg = read_config(cfg)
    # default case: create env_num envs with the copy of env cfg.
    # if you want to indicate different cfg for different env, please refer to `get_vec_env_setting`.
    # usually, user defined env must be registered in nervex so that it can be created with config string,
    # and you can also directly pass env_fn argument, in some dynamic env class cases.
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
    # seed
    actor_env.seed(seed)
    evaluator_env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cfg.policy.use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # create component
    policy_fn = create_policy if policy_type is None else policy_type
    policy = policy_fn(cfg.policy, model=model)
    learner = BaseLearner(cfg.learner)
    actor = BaseSerialActor(cfg.actor)
    evaluator = BaseSerialEvaluator(cfg.evaluator)
    replay_buffer = BufferManager(cfg.replay_buffer)
    command = BaseSerialCommand(cfg.command, learner, actor, evaluator, replay_buffer)

    actor.env = actor_env
    evaluator.env = evaluator_env
    learner.policy = policy.learn_mode
    actor.policy = policy.collect_mode
    evaluator.policy = policy.eval_mode
    command.policy = policy.command_mode
    # main loop
    max_eval_reward = float("-inf")
    learner_train_step = cfg.policy.learn.train_step
    # Here we assume serial entry mainly focuses on agent buffer.
    # ``enough_data_count``` is just a lower bound estimation. It is possible that replay buffer's data count is
    # greater than this value, but still has no enough data to train ``train_step`` times.
    enough_data_count = cfg.policy.learn.batch_size * max(
        cfg.replay_buffer.agent.min_sample_ratio,
        math.ceil(cfg.policy.learn.train_step / cfg.replay_buffer.agent.max_reuse)
    )
    use_priority = cfg.policy.get('use_priority', False)
    learner.call_hook('before_run')
    while True:
        command.step()
        while True:
            # actor keeps generating data until replay buffer has enough to sample one batch
            new_data, collect_info = actor.generate_data(learner.train_iter)
            replay_buffer.push_data(new_data)
            if replay_buffer.count() >= enough_data_count:
                break
        learner.collect_info = collect_info
        for i in range(learner_train_step):
            # Learner will train ``train_step`` times in one iteration.
            # But if replay buffer does not have enough data, program will break and warn.
            train_data = replay_buffer.sample(cfg.policy.learn.batch_size, learner.train_iter)
            if train_data is None:
                warnings.warn(
                    "Replay buffer's data can only train for {} steps. ".format(i) +
                    "You can modify data collect config, e.g. increasing n_sample, n_episode or min_sample_ratio."
                )
                break
            learner.train(train_data)
            if use_priority:
                replay_buffer.update(learner.priority_info)
        if (learner.train_iter - 1) % cfg.evaluator.eval_freq == 0:
            stop_flag, eval_reward = evaluator.eval(learner.train_iter)
            if stop_flag:
                # evaluator's mean episode reward reaches the expected ``stop_val``
                learner.save_checkpoint()
                print("Your RL agent is converged, you can refer to 'log/evaluator/evaluator_logger.txt' for details")
                break
            else:
                if eval_reward > max_eval_reward:
                    learner.save_checkpoint()
                    max_eval_reward = eval_reward
        if cfg.policy.on_policy:
            replay_buffer.clear()
    learner.call_hook('after_run')

    # close
    replay_buffer.close()
    learner.close()
    actor.close()
    evaluator.close()


def eval(
        cfg: Union[str, dict],
        seed: int,
        env_setting: Optional[Any] = None,  # subclass of BaseEnv, and config dict
        policy_type: Optional[type] = None,  # subclass of Policy
        model: Optional[Union[type, torch.nn.Module]] = None,  # instance or subclass of torch.nn.Module
) -> None:
    if isinstance(cfg, str):
        cfg = read_config(cfg)
    manager_cfg = cfg.env.get('manager', {})
    if env_setting is None:
        env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, _, evaluator_env_cfg = env_setting
    env_manager_type = BaseEnvManager if cfg.env.env_manager_type == 'base' else SubprocessEnvManager
    evaluator_env = env_manager_type(
        env_fn, env_cfg=evaluator_env_cfg, env_num=len(evaluator_env_cfg), manager_cfg=manager_cfg
    )
    # seed
    evaluator_env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cfg.policy.use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # create component
    policy_fn = create_policy if policy_type is None else policy_type
    policy = policy_fn(cfg.policy, model=model, enable_field=['eval'])
    state_dict = torch.load(cfg.learner.load_path, map_location='cpu')
    policy.state_dict_handle()['model'].load_state_dict(state_dict['model'])
    evaluator = BaseSerialEvaluator(cfg.evaluator)

    evaluator.env = evaluator_env
    evaluator.policy = policy.eval_mode
    # eval
    _, eval_reward = evaluator.eval(0)
    print('Eval is over! The performance of your RL policy is {}'.format(eval_reward))
