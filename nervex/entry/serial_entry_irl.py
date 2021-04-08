from typing import Union, Optional, List, Any
import torch
import logging

from nervex.envs import get_vec_env_setting, create_env_manager
from nervex.worker import BaseLearner, BaseSerialCollector, BaseSerialEvaluator, BaseSerialCommander
from nervex.config import read_config
from nervex.data import BufferManager
from nervex.policy import create_policy
from nervex.utils import build_logger
from nervex.irl_utils import create_irl_model
from .utils import set_pkg_seed


def serial_pipeline_irl(
        cfg: Union[str, dict],
        seed: int = None,
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
        collector_log = logging.getLogger('collector_logger')
        collector_log.disabled = True
    if isinstance(cfg, str):
        cfg = read_config(cfg)
    # Default case: Create env_num envs with copies of env cfg.
    # If you want to indicate different cfg for different env, please refer to ``get_vec_env_setting``.
    # Usually, user-defined env must be registered in nervex so that it can be created with config string;
    # Or you can also directly pass in env_fn argument, in some dynamic env class cases.
    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    manager_cfg = cfg.env.get('manager', {})
    collector_env = create_env_manager(
        cfg.env.env_manager_type,
        env_fn=env_fn,
        env_cfg=collector_env_cfg,
        env_num=len(collector_env_cfg),
        manager_cfg=manager_cfg
    )
    evaluator_env = create_env_manager(
        cfg.env.env_manager_type,
        env_fn=env_fn,
        env_cfg=evaluator_env_cfg,
        env_num=len(evaluator_env_cfg),
        manager_cfg=manager_cfg
    )
    # Random seed
    if not seed:
        seed = cfg.get('seed', 0)
    collector_env.seed(seed)
    evaluator_env.seed(seed)
    set_pkg_seed(seed, use_cuda=cfg.policy.use_cuda)
    # Create components: policy, learner, collector, evaluator, replay buffer, commander.
    if policy_type is None:
        policy_fn = create_policy
    else:
        assert callable(policy_type)
        policy_fn = policy_type
    policy = policy_fn(cfg.policy, model=model)
    _, tb_logger = build_logger(path='./log/', name='serial', need_tb=True, need_text=False)
    learner = BaseLearner(cfg.learner, tb_logger)
    collector = BaseSerialCollector(cfg.collector, tb_logger)
    evaluator = BaseSerialEvaluator(cfg.evaluator, tb_logger)
    replay_buffer = BufferManager(cfg.replay_buffer, tb_logger)
    commander = BaseSerialCommander(cfg.commander, learner, collector, evaluator, replay_buffer)
    reward_model = create_irl_model(cfg.irl, policy.get_attribute('device'))
    reward_model.start()
    # Set corresponding env and policy mode.
    collector.env = collector_env
    evaluator.env = evaluator_env
    learner.policy = policy.learn_mode
    collector.policy = policy.collect_mode
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
    # How many steps to train in collector's one collection.
    learner_train_iteration = cfg.policy.learn.train_iteration
    # Whether to switch on priority experience replay.
    use_priority = cfg.policy.get('use_priority', False)
    # Learner's before_run hook.
    learner.call_hook('before_run')

    # Accumulate plenty of data at the beginning of training.
    if replay_buffer.replay_start_size() > 0:
        new_data = collector.generate_data(learner.train_iter, n_sample=replay_buffer.replay_start_size())
        replay_buffer.push(new_data, cur_collector_envstep=0)

    while True:
        commander.step()
        # Evaluate at the beginning of training.
        if eval_interval >= cfg.evaluator.eval_freq:
            stop_flag, eval_reward = evaluator.eval(learner.train_iter, collector.envstep)
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
        new_data_count, target_new_data_count = 0, cfg.irl.get('target_new_data_count', 1)
        while new_data_count < target_new_data_count:
            new_data = collector.generate_data(learner.train_iter)
            new_data_count += len(new_data)
            # collect data for reward_model training
            reward_model.collect_data(new_data)
            replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # update reward_model
        reward_model.train()
        reward_model.clear_data()
        # Learn policy from collected data
        for i in range(learner_train_iteration):
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
            # update train_data reward
            reward_model.estimate(train_data)
            learner.train(train_data, collector.envstep)
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
    collector.close()
    evaluator.close()
    return policy
