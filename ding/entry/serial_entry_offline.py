from typing import Union, Optional, List, Any, Tuple
import os
import torch
from tqdm import tqdm
from functools import partial
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.utils.data import create_dataset, default_collate


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
        cfg, create_cfg = input_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

    # Dataset
    dataset = create_dataset(cfg)
    if 'holdout_ratio' not in cfg.policy.learn:
        dataloader = DataLoader(
            dataset, 
            cfg.policy.learn.batch_size, 
            shuffle=True, 
            collate_fn=lambda x: x,
            pin_memory=cfg.policy.cuda,
        )
    else:
        # holdout validation set applies in behavioral cloning
        ratio = cfg.policy.learn.holdout_ratio
        split = -int(len(dataset)*ratio)
        dataloader = DataLoader(
            dataset[:split], 
            cfg.policy.learn.batch_size, 
            shuffle=True, 
            collate_fn=lambda x: x,
            pin_memory=cfg.policy.cuda,
        )
        eval_dataloader = DataLoader(
            dataset[split:], 
            cfg.policy.learn.batch_size, 
            shuffle=True, 
            collate_fn=default_collate,
            pin_memory=cfg.policy.cuda,
        )
    # Env, Policy
    env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env, collect=False)
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    # Random seed
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'eval'])

    # Normalization for state in offlineRL dataset.
    if cfg.policy.collect.get('normalize_states', None):
        policy.set_norm_statistics(dataset.statistics)

    # Main components
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
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

    for epoch in tqdm(range(cfg.policy.learn.train_epoch)):
        # Evaluate policy per epoch
        for train_data in tqdm(dataloader):
            learner.train(train_data)

        if 'holdout_ratio' in cfg.policy.learn:
            loss_list = []
            for eval_data in eval_dataloader:
                res = policy._forward_eval(eval_data['obs'])
                loss_list.append(torch.nn.MSELoss()(res['action'], eval_data['action'].squeeze(-1)).item())
            tb_logger.add_scalar('validation_mse', sum(loss_list) / len(loss_list), epoch)
            
        stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter)

        if stop or learner.train_iter >= max_train_iter:
            stop = True
            break

    learner.call_hook('after_run')
    print('final reward is: {}'.format(reward))
    return policy, stop
