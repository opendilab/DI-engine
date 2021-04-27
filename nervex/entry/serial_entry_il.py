from typing import Union, Optional, List, Any, Callable
import os
import torch
import logging
from functools import partial
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from nervex.worker import BaseLearner, BaseSerialEvaluator
from nervex.envs import get_vec_env_setting, create_env_manager
from nervex.config import read_config
from nervex.policy import create_policy
from nervex.data import NaiveRLDataset
from .utils import set_pkg_seed


def serial_pipeline_il(
        cfg: Union[str, dict],
        seed: int,
        data_path: str,
        model: Optional[torch.nn.Module] = None,
) -> Union['BasePolicy', bool]:  # noqa
    r"""
    Overview:
        Serial pipeline entry.
    Arguments:
        - cfg (:obj:`Union[str, dict]`): Config in dict type. ``str`` type means config file path.
        - seed (:obj:`int`): Random seed.
    """
    if isinstance(cfg, str):
        cfg = read_config(cfg)
    env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env.env_kwargs)
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    # Random seed
    evaluator_env.seed(seed)
    set_pkg_seed(seed, use_cuda=cfg.policy.use_cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'eval'])
    tb_logger = SummaryWriter(os.path.join('./log/', 'serial_il'))
    dataset = NaiveRLDataset(data_path)
    dataloader = DataLoader(dataset, cfg.policy.learn.batch_size, collate_fn=lambda x: x)
    learner = BaseLearner(cfg.learner, policy.learn_mode, tb_logger)
    evaluator = BaseSerialEvaluator(cfg.evaluator, evaluator_env, policy.eval_mode, tb_logger)
    # ==========
    # Main loop
    # ==========
    learner.call_hook('before_run')
    stop = False

    for epoch in range(cfg.policy.learn.train_epoch):
        # Evaluate policy performance
        for i, train_data in enumerate(dataloader):
            if evaluator.should_eval(learner.train_iter):
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter)
                if stop:
                    break
            learner.train(train_data)
        if stop:
            break

    learner.call_hook('after_run')
    print('final reward is: {}'.format(reward))
    return policy, stop
