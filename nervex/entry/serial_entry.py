from typing import Union, Optional, List, Any, Callable
import torch
import logging

from nervex.envs import get_vec_env_setting, create_env_manager
from nervex.worker import BaseLearner, BaseSerialActor, BaseSerialEvaluator, BaseSerialCommander
from nervex.config import read_config
from nervex.data import BufferManager
from nervex.policy import create_policy
from nervex.utils import build_logger
from .utils import set_pkg_seed


def serial_pipeline(
        cfg: Union[str, dict],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
) -> 'BasePolicy':  # noqa
    r"""
    Overview:
        Serial pipeline entry.
    Arguments:
        - cfg (:obj:`Union[str, dict]`): Config in dict type. ``str`` type means config file path.
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): Subclass of ``BaseEnv``, and config dict.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
    """
    if isinstance(cfg, str):
        cfg = read_config(cfg)
    if env_setting is None:
        env_fn, actor_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env.env_kwargs)
    else:
        env_fn, actor_env_cfg, evaluator_env_cfg = env_setting
    actor_env = create_env_manager(cfg.env.manager, env_fn, actor_env_cfg)
    evaluator_env = create_env_manager(cfg.env.manager, env_fn, evaluator_env_cfg)
    # Random seed
    actor_env.seed(seed)
    evaluator_env.seed(seed)
    set_pkg_seed(seed, use_cuda=cfg.policy.use_cuda)
    # Create components: policy, learner, actor, evaluator, replay buffer, commander.
    policy = create_policy(cfg.policy, model=model)
    _, tb_logger = build_logger(path='./log/', name='serial', need_tb=True, need_text=False)
    learner = BaseLearner(cfg.learner, tb_logger)
    actor = BaseSerialActor(cfg.actor, actor_env, tb_logger)
    evaluator = BaseSerialEvaluator(cfg.evaluator, evaluator_env, tb_logger)
    replay_buffer = BufferManager(cfg.replay_buffer, tb_logger)
    commander = BaseSerialCommander(cfg.commander, learner, actor, evaluator, replay_buffer)
    # Set corresponding policy mode.
    learner.policy = policy.learn_mode
    actor.policy = policy.collect_mode
    evaluator.policy = policy.eval_mode
    commander.policy = policy.command_mode
    # ==========
    # Main loop
    # ==========
    max_eval_reward = float("-inf")
    eval_interval = cfg.evaluator.eval_freq
    use_priority = cfg.policy.get('use_priority', False)

    # Learner's before_run hook.
    learner.call_hook('before_run')

    # Accumulate plenty of data at the beginning of training.
    if replay_buffer.replay_start_size() > 0:
        commander.step()
        new_data = actor.generate_data(learner.train_iter, n_sample=replay_buffer.replay_start_size())
        replay_buffer.push(new_data, cur_actor_envstep=0)

    while True:
        commander.step()
        # Evaluate at the beginning of training.
        if eval_interval >= cfg.evaluator.eval_freq:
            stop_flag, eval_reward = evaluator.eval(learner.train_iter, actor.envstep)
            eval_interval = 0
            if stop_flag and learner.train_iter > 0:
                # Evaluator's mean episode reward reaches the expected ``stop_value``.
                print(
                    "[nerveX serial pipeline] Your RL agent is converged, you can refer to " +
                    "'log/evaluator/evaluator_logger.txt' for details"
                )
                break
            else:
                if eval_reward > max_eval_reward:
                    learner.save_checkpoint('ckpt_best.pth.tar')
                    max_eval_reward = eval_reward
        # Collect data by default config n_sample/n_episode
        new_data = actor.generate_data(learner.train_iter)
        replay_buffer.push(new_data, cur_actor_envstep=actor.envstep)
        # TODO whether adjust train_iteration by the number of the collected data
        # Learn policy from collected data
        for i in range(cfg.policy.learn.train_iteration):
            # Learner will train ``train_iteration`` times in one iteration.
            # But if replay buffer does not have enough data, program will break and warn.
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is None:
                # It is possible that replay buffer's data count is too few to train ``train_iteration`` times
                logging.warning(
                    "Replay buffer's data can only train for {} steps. ".format(i) +
                    "You can modify data collect config, e.g. increasing n_sample, n_episode."
                )
                break
            learner.train(train_data, actor.envstep)
            eval_interval += 1
            if use_priority:
                replay_buffer.update(learner.priority_info)
        if cfg.policy.on_policy:
            # On-policy algorithm must clear the replay buffer.
            replay_buffer.clear()
    # Learner's after_run hook.
    learner.call_hook('after_run')
    # Close all resources.
    learner.close()
    actor.close()
    evaluator.close()
    return policy
