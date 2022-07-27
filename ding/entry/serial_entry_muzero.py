from typing import Union, Optional, List, Any, Tuple
import os
import torch
import logging
from functools import partial
from tensorboardX import SummaryWriter
import numpy as np
from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, BaseSerialCommander, create_serial_collector
from ding.worker.collector.muzero_evaluator import MuZeroEvaluator as BaseSerialEvaluator

from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.rl_utils.mcts.game_buffer import GameBuffer


# @profile
def serial_pipeline_muzero(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
        game_config: Optional[dict] = None,
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry for MuZero and its variants, such as EfficientZero.
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
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])

    # load pretrained model
    if cfg.policy.model_path is not None:
        policy.learn_mode.load_state_dict(torch.load(cfg.policy.model_path, map_location='cpu'))

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    # MuZero related code
    # specific game buffer for MuZero
    replay_buffer = GameBuffer(game_config)
    collector = create_serial_collector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        replay_buffer=replay_buffer,
        game_config=game_config
    )
    evaluator = BaseSerialEvaluator(
        cfg.policy.eval.evaluator,
        evaluator_env,
        policy.eval_mode,
        tb_logger,
        exp_name=cfg.exp_name,
        game_config=game_config
    )

    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode
    )
    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')

    while True:
        collect_kwargs = commander.step()
        # set temperature for visit count distributions according to the train_iter,
        # please refer to Appendix A.1 in EfficientZero for details
        collect_kwargs['temperature'] = np.array(
            [
                game_config.visit_count_temperature(trained_steps=learner.train_iter)
                for _ in range(game_config.collector_env_num)
            ]
        )

        # TODO(pu): eval trained model
        # returns = []
        # test_episodes = 1
        # for i in range(test_episodes):
        #     stop, reward = evaluator.eval(
        #         learner.save_checkpoint, learner.train_iter, collector.envstep, config=game_config
        #     )
        #     returns.append(reward)
        # print(returns)
        # returns = np.array(returns)
        # print(f'win rate: {len(np.where(returns == 1.)[0])/ test_episodes}, draw rate: {len(np.where(returns == 0.)[0])/test_episodes},
        # lose rate: {len(np.where(returns == -1.)[0])/ test_episodes}')
        # break

        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(
                learner.save_checkpoint, learner.train_iter, collector.envstep, config=game_config
            )
            if stop:
                break

        # Collect data by default config n_sample/n_episode
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        # remove the oldest data if the replay buffer is full.
        replay_buffer.remove_oldest_data_to_fit()
        # TODO(pu): collector return data
        # replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # Learn policy from collected data
        for i in range(cfg.policy.learn.update_per_collect):
            # Learner will train ``update_per_collect`` times in one iteration.
            try:
                train_data = replay_buffer.sample_train_data(learner.policy.get_attribute('batch_size'), policy)
            except Exception as exception:
                print(exception)
                logging.warning(
                    f'The data in replay_buffer is not sufficient to sample a minibatch: '
                    f'batch_size: {replay_buffer.get_batch_size()},'
                    f'num_of_episodes: {replay_buffer.get_num_of_episodes()}, '
                    f'num of game historys: {replay_buffer.get_num_of_game_histories()}, '
                    f'number of transitions: {replay_buffer.get_num_of_transitions()}, '
                    f'continue to collect now ....'
                )
                break

            learner.train(train_data, collector.envstep)

            # if game_config.lr_manually:
            #     # learning rate decay manually like EfficientZero paper
            #     if learner.train_iter > 1e5 and learner.train_iter <= 2e5:
            #         policy._optimizer.lr = 0.02
            #     elif learner.train_iter > 2e5:
            #         policy._optimizer.lr = 0.002
            if game_config.lr_manually:
                if learner.train_iter < 0.5 * game_config.max_training_steps:
                    policy._optimizer.lr = 0.2
                elif learner.train_iter < 0.75 * game_config.max_training_steps:
                    policy._optimizer.lr = 0.02
                else:
                    policy._optimizer.lr = 0.002

        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy
