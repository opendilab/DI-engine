import sys
import copy
import time
from typing import Union, Optional, List, Any
import numpy as np
import torch

from nervex.worker import BaseLearner, BaseSerialActor, BaseSerialEvaluator, BaseSerialCommand
from nervex.worker import BaseEnvManager, SubprocessEnvManager
from nervex.utils import read_config
from nervex.data import ReplayBuffer
from nervex.policy import create_policy
from nervex.envs import get_vec_env_setting


def serial_pipeline(
        cfg: Union[str, dict],
        seed: int,
        env_setting: Optional[Any] = None,  # subclass of BaseEnv, and config dict
        policy_type: Optional[type] = None,  # subclass of Policy
        model_type: Optional[type] = None,  # subclass of torch.nn.Module
) -> None:
    if isinstance(cfg, str):
        cfg = read_config(cfg)
    if env_setting is None:
        env_fn, actor_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, actor_env_cfg, evaluator_env_cfg = env_setting
    env_manager_type = BaseEnvManager if cfg.env.env_manager_type == 'base' else SubprocessEnvManager
    actor_env = env_manager_type(env_fn=env_fn, env_cfg=actor_env_cfg, env_num=len(actor_env_cfg))
    evaluator_env = env_manager_type(env_fn, env_cfg=evaluator_env_cfg, env_num=len(evaluator_env_cfg))
    # seed
    actor_env.seed(seed)
    evaluator_env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # create component
    policy_fn = create_policy if policy_type is None else policy_type
    policy = policy_fn(cfg.policy, model_type)
    learner = BaseLearner(cfg)
    actor = BaseSerialActor(cfg.actor)
    evaluator = BaseSerialEvaluator(cfg.evaluator)
    replay_buffer = ReplayBuffer(cfg.replay_buffer)
    command = BaseSerialCommand(cfg.command, learner, actor, evaluator, replay_buffer)

    actor.env = actor_env
    evaluator.env = evaluator_env
    learner.policy = policy.learn_mode
    actor.policy = policy.collect_mode
    evaluator.policy = policy.eval_mode
    command.policy = policy.command_mode
    learner.launch()
    # main loop
    iter_count = 0
    while True:
        command.step()
        while True:
            new_data, collect_info = actor.generate_data()
            replay_buffer.push_data(new_data)
            if replay_buffer.count >= cfg.policy.learn.batch_size * cfg.replay_buffer.min_sample_ratio:
                break
        learner.collect_info = collect_info
        for _ in range(cfg.policy.learn.train_step):
            train_data = replay_buffer.sample(cfg.policy.learn.batch_size)
            learner.train(train_data)
        if iter_count % cfg.evaluator.eval_freq == 0 and evaluator.eval(iter_count * cfg.policy.learn.train_step):
            learner.save_checkpoint()
            print("Your RL agent is converged, you can refer to 'log/evaluator.txt' for details")
            break
        if cfg.policy.on_policy:
            replay_buffer.clear()
        iter_count += 1

    # close
    replay_buffer.close()
    learner.close()
    actor.close()
    evaluator.close()
