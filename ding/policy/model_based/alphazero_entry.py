import logging
import os
from typing import Union, Optional, Tuple

import torch
from tensorboardX import SummaryWriter

from ding.config.config import read_config_yaml
from ding.envs import get_env_cls
from ding.policy.model_based.alphazero_policy import AlphaZeroPolicy
from ding.model import create_model
from ding.policy.model_based.alphazero_collector_ding import AlphazeroCollector
from ding.policy.model_based.alphazero_evaluator_ding import AlphazeroEvaluator
from ding.utils import set_pkg_seed
from ding.worker import BaseLearner, create_buffer


def serial_alphazero_pipeline(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        # env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_iterations: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_iterations (:obj:`Optional[torch.nn.Module]`): Learner's max iteration. Pipeline will stop \
            when reaching this iteration.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """

    # ============
    # Setup Config
    # ============
    if isinstance(input_cfg, str):
        cfg = read_config_yaml(input_cfg)
    else:
        cfg = input_cfg

    # =========
    # Setup Env
    # =========
    env_fn = get_env_cls(cfg.env)
    collector_env = env_fn(cfg.env)
    evaluator_env = env_fn(cfg.env)

    # ==========
    # Setup Seed
    # ==========
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    # ============
    # Setup Policy
    # ============
    model = model if model else create_model(cfg.model)
    policy = AlphaZeroPolicy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', ])

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = AlphazeroCollector(
        cfg=cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )
    evaluator = AlphazeroEvaluator(
        cfg=cfg.policy.eval.evaluator,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )

    replay_buffer = create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
    # commander = BaseSerialCommander(
    #     cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode
    # )
    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')

    # Accumulate plenty of data at the beginning of training.
    for _ in range(max_iterations):
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval()
            if stop:
                break
        # Collect data by default config n_sample/n_episode
        new_data = collector.collect(train_iter=learner.train_iter)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
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
            learner.train(train_data, collector.envstep)

    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy


if __name__ == '__main__':
    cfg_path = os.path.join(os.getcwd(), 'alphazero_config_ding.yaml')
    serial_alphazero_pipeline(cfg_path)
