from typing import Union, Optional, Tuple
import os
import torch
from functools import partial
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.utils.data import NaiveRLDataset


def serial_pipeline_il(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int,
        data_path: str,
        model: Optional[torch.nn.Module] = None,
) -> Union['Policy', bool]:  # noqa
    r"""
    Overview:
        Serial pipeline entry of imitation learning.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - data_path (:obj:`str`): Path of training data.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
        - convergence (:obj:`bool`): whether il training is converged
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

    # Env, Policy
    env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    # Random seed
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'eval'])

    # Main components
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    dataset = NaiveRLDataset(data_path)
    dataloader = DataLoader(dataset, cfg.policy.learn.batch_size, collate_fn=lambda x: x)
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
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
