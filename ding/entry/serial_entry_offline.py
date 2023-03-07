from typing import Union, Optional, List, Any, Tuple
import os
import torch
from functools import partial
from tensorboardX import SummaryWriter
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed, get_world_size, get_rank
from ding.utils.data import create_dataset


def serial_pipeline_offline(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
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
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

    # Dataset
    dataset = create_dataset(cfg)
    sampler, shuffle = None, True
    if get_world_size() > 1:
        sampler, shuffle = DistributedSampler(dataset), False
    dataloader = DataLoader(
        dataset,
        # Dividing by get_world_size() here simply to make multigpu
        # settings mathmatically equivalent to the singlegpu setting.
        # If the training efficiency is the bottleneck, feel free to
        # use the original batch size per gpu and increase learning rate
        # correspondingly.
        cfg.policy.learn.batch_size // get_world_size(),
        # cfg.policy.learn.batch_size
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=lambda x: x,
        pin_memory=cfg.policy.cuda,
    )
    # Env, Policy
    try:
        if cfg.env.norm_obs.use_norm and cfg.env.norm_obs.offline_stats.use_offline_stats:
            cfg.env.norm_obs.offline_stats.update({'mean': dataset.mean, 'std': dataset.std})
    except (KeyError, AttributeError):
        pass
    env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env, collect=False)
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    # Random seed
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'eval'])

    if hasattr(policy, 'set_statistic'):
        # useful for setting action bounds for ibc
        policy.set_statistic(dataset.statistics)

    # Otherwise, directory may conflicts in the multigpu settings.
    if get_rank() == 0:
        tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    else:
        tb_logger = None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')
    stop = False

    for epoch in range(cfg.policy.learn.train_epoch):
        if get_world_size() > 1:
            dataloader.sampler.set_epoch(epoch)
        for train_data in dataloader:
            learner.train(train_data)

        # Evaluate policy at most once per epoch.
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter)

        if stop or learner.train_iter >= max_train_iter:
            stop = True
            break

    learner.call_hook('after_run')
    print('final reward is: {}'.format(reward))
    return policy, stop
