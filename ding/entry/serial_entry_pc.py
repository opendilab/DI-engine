from typing import Union, Optional, Tuple
import os
from functools import partial
from copy import deepcopy

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.utils.data.dataset import load_bfs_datasets


def serial_pipeline_pc(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        max_iter=int(1e6),
) -> Union['Policy', bool]:  # noqa
    r"""
    Overview:
        Serial pipeline entry of procedure cloning using BFS as expert policy.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_iter (:obj:`Optional[int]`): Max iteration for executing PC training.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
        - convergence (:obj:`bool`): whether the training is converged
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
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
    train_data, test_data = load_bfs_datasets(train_seeds=cfg.train_seeds)
    dataloader = DataLoader(train_data, batch_size=cfg.policy.learn.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=cfg.policy.learn.batch_size, shuffle=True)
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )

    # ==========
    # Main loop
    # ==========
    learner.call_hook('before_run')
    stop = False
    iter_cnt = 0
    for epoch in range(cfg.policy.learn.train_epoch):
        # train
        criterion = torch.nn.CrossEntropyLoss()
        for i, train_data in enumerate(dataloader):
            learner.train(train_data)
            iter_cnt += 1
            if iter_cnt >= max_iter:
                stop = True
                break
        if epoch % 69 == 0:
            policy._optimizer.param_groups[0]['lr'] /= 10
        if stop:
            break
        losses = []
        acces = []
        # Evaluation
        for _, test_data in enumerate(test_dataloader):
            observations, bfs_input_maps, bfs_output_maps = test_data['obs'], test_data['bfs_in'].long(), \
                                                            test_data['bfs_out'].long()
            states = observations
            bfs_input_onehot = torch.nn.functional.one_hot(bfs_input_maps, 5).float()

            bfs_states = torch.cat([
                states,
                bfs_input_onehot,
            ], dim=-1).cuda()
            logits = policy._model(bfs_states)['logit']
            logits = logits.flatten(0, -2)
            labels = bfs_output_maps.flatten(0, -1).cuda()

            loss = criterion(logits, labels).item()
            preds = torch.argmax(logits, dim=-1)
            acc = torch.sum((preds == labels)) / preds.shape[0]

            losses.append(loss)
            acces.append(acc)
        print('Test Finished! Loss: {} acc: {}'.format(sum(losses) / len(losses), sum(acces) / len(acces)))
    stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter)
    learner.call_hook('after_run')
    print('final reward is: {}'.format(reward))
    return policy, stop
