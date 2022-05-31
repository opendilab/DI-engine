from typing import Union, Optional, List, Any, Tuple
import torch

from ding.worker import create_buffer
from ding.entry.utils import random_collect
from ding.world_model.entry.utils import mbrl_entry_setup


def serial_pipeline_dyna(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry for dyna-style model-based RL.
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
    cfg, policy, world_model, env_buffer, learner, collector, collector_env, evaluator, commander, tb_logger = \
        mbrl_entry_setup(input_cfg, seed, env_setting, model)

    # dyna-style algorithm maintains a imaginiation buffer from model rollouts
    img_buffer_cfg = cfg.world_model.other.imagination_buffer
    if img_buffer_cfg.type == 'elastic':
        img_buffer_cfg.set_buffer_size = world_model.buffer_size_scheduler
    img_buffer = create_buffer(cfg.world_model.other.imagination_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)

    learner.call_hook('before_run')

    if cfg.policy.get('random_collect_size', 0) > 0:
        random_collect(cfg.policy, policy, collector, collector_env, commander, env_buffer)

    while True:
        collect_kwargs = commander.step()
        # eval the policy
        if evaluator.should_eval(collector.envstep):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break

        # fill environment buffer
        data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        env_buffer.push(data, cur_collector_envstep=collector.envstep)

        # eval&train world model and fill imagination buffer
        if world_model.should_eval(collector.envstep):
            world_model.eval(env_buffer, collector.envstep, learner.train_iter)
        if world_model.should_train(collector.envstep):
            world_model.train(env_buffer, collector.envstep, learner.train_iter)
            world_model.fill_img_buffer(
                policy.collect_mode, env_buffer, img_buffer, collector.envstep, learner.train_iter
            )

        for i in range(cfg.policy.learn.update_per_collect):
            batch_size = learner.policy.get_attribute('batch_size')
            train_data = world_model.sample(env_buffer, img_buffer, batch_size, learner.train_iter)
            learner.train(train_data, collector.envstep)

        if cfg.policy.on_policy:
            # On-policy algorithm must clear the replay buffer.
            env_buffer.clear()
            img_buffer.clear()

        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    learner.call_hook('after_run')

    return policy
