from typing import Optional, Tuple
import os
import logging
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.config import compile_config
from ding.policy import create_policy, PolicyFactory
from ding.reward_model import create_reward_model
from ding.utils import set_pkg_seed
from ding.entry import collect_demo_data
from dizoo.classic_control.cartpole.config.cartpole_gail_config import cartpole_gail_config, cartpole_gail_create_config
from dizoo.classic_control.cartpole.config.cartpole_dqn_config import cartpole_dqn_config, cartpole_dqn_create_config
from ding.utils import save_file


def main(
        input_cfg: Tuple[dict, dict],
        expert_cfg: Tuple[dict, dict],
        seed: int = 0,
        max_iterations: Optional[int] = int(1e4),
        collect_data: bool = True,
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry with reward model.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - max_iterations (:obj:`Optional[torch.nn.Module]`): Learner's max iteration. Pipeline will stop \
            when reaching this iteration.
        - collect_data (:obj:`bool`): Collect expert data.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    cfg, create_cfg = input_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg, save_cfg=True)
    expert_cfg, expert_create_cfg = expert_cfg
    # Load expert data
    if collect_data:
        expert_cfg.policy.other.eps.collect = 0.1
        expert_cfg.policy.load_path = os.path.join(expert_cfg.exp_name, 'ckpt/ckpt_best.pth.tar')
        collect_demo_data((expert_cfg, expert_create_cfg), seed, state_dict_path=expert_cfg.policy.load_path,
                          expert_data_path=cfg.reward_model.expert_data_path,
                          collect_count=cfg.reward_model.collect_count)
    # Create main components: env, policy
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, enable_field=['learn', 'collect', 'eval', 'command'])

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
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
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
        action_space = collector_env.env_info().act_space
        random_policy = PolicyFactory.get_random_policy(policy.collect_mode, action_space=action_space)
        collector.reset_policy(random_policy)
        collect_kwargs = commander.step()
        new_data = collector.collect(n_sample=cfg.policy.random_collect_size, policy_kwargs=collect_kwargs)
        replay_buffer.push(new_data, cur_collector_envstep=0)
        collector.reset_policy(policy.collect_mode)
    for _ in range(max_iterations):
        collect_kwargs = commander.step()
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        new_data_count, target_new_data_count = 0, cfg.reward_model.get('target_new_data_count', 1)
        while new_data_count < target_new_data_count:
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
            new_data_count += len(new_data)
            # collect data for reward_model training
            reward_model.collect_data(new_data)
            replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # update reward_model
        reward_model.train()
        reward_model.clear_data()
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
            # update train_data reward
            reward_model.estimate(train_data)
            learner.train(train_data, collector.envstep)
            if learner.policy.get_attribute('priority'):
                replay_buffer.update(learner.priority_info)
        if cfg.policy.on_policy:
            # On-policy algorithm must clear the replay buffer.
            replay_buffer.clear()

    # Learner's after_run hook.
    learner.call_hook('after_run')
    # save reward model
    path = os.path.join(cfg.exp_name, 'reward_model', 'ckpt')
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
    path = os.path.join(path, 'last.pth.tar')
    state_dict = reward_model.state_dict()
    save_file(path, state_dict)
    print('Saved reward model ckpt in {}'.format(path))
    # evaluate
    evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)


if __name__ == "__main__":
    main((cartpole_gail_config, cartpole_gail_create_config), (cartpole_dqn_config, cartpole_dqn_create_config),
         collect_data=0, seed=0)
