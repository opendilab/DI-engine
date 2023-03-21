from typing import Union, Optional, Tuple
import os
import torch
from functools import partial
from tensorboardX import SummaryWriter
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
import pickle

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed


class MCTSPCDataset(Dataset):

    def __init__(self, data_dic, seq_len=4):
        self.observations = data_dic['obs']
        self.actions = data_dic['actions']
        self.hidden_states = data_dic['hidden_state']
        self.seq_len = seq_len
        self.length = len(self.observations) - seq_len - 1

    def __getitem__(self, idx):
        """
        Assume the trajectory is: o1, h2, h3, h4
        """
        return {
            'obs': self.observations[idx],
            'hidden_states': list(reversed(self.hidden_states[idx + 1:idx + self.seq_len + 1])),
            'action': self.actions[idx]
        }

    def __len__(self):
        return self.length


def load_mcts_datasets(path, seq_len, batch_size=32):
    with open(path, 'rb') as f:
        dic = pickle.load(f)
    tot_len = len(dic['obs'])
    train_dic = {k: v[:-tot_len // 10] for k, v in dic.items()}
    test_dic = {k: v[-tot_len // 10:] for k, v in dic.items()}
    return DataLoader(MCTSPCDataset(train_dic, seq_len=seq_len), shuffle=True, batch_size=batch_size), \
        DataLoader(MCTSPCDataset(test_dic, seq_len=seq_len), shuffle=True, batch_size=batch_size)


def serial_pipeline_pc_mcts(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        max_iter=int(1e6),
) -> Union['Policy', bool]:  # noqa
    r"""
    Overview:
        Serial pipeline entry of procedure cloning with MCTS.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
        - convergence (:obj:`bool`): whether il training is converged
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
    dataloader, test_dataloader = load_mcts_datasets(cfg.policy.expert_data_path, seq_len=cfg.policy.seq_len,
                                                     batch_size=cfg.policy.learn.batch_size)
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )

    # ==========
    # Main loop
    # ==========
    learner.call_hook('before_run')
    stop = False
    epoch_per_test = 10
    for epoch in range(cfg.policy.learn.train_epoch):
        # train
        criterion = torch.nn.CrossEntropyLoss()
        for i, train_data in enumerate(dataloader):
            train_data['obs'] = train_data['obs'].permute(0, 3, 1, 2).float().cuda() / 255.
            learner.train(train_data)
            if learner.train_iter >= max_iter:
                stop = True
                break
        if epoch % 69 == 0:
            policy._optimizer.param_groups[0]['lr'] /= 10
        if stop:
            break

        if epoch % epoch_per_test == 0:
            losses = []
            acces = []
            for _, test_data in enumerate(test_dataloader):
                logits = policy._model.forward_eval(test_data['obs'].permute(0, 3, 1, 2).float().cuda() / 255.)
                loss = criterion(logits, test_data['action'].cuda()).item()
                preds = torch.argmax(logits, dim=-1)
                acc = torch.sum((preds == test_data['action'].cuda())).item() / preds.shape[0]

                losses.append(loss)
                acces.append(acc)
            tb_logger.add_scalar('learn_iter/recurrent_test_loss', sum(losses) / len(losses), learner.train_iter)
            tb_logger.add_scalar('learn_iter/recurrent_test_acc', sum(acces) / len(acces), learner.train_iter)

            losses = []
            acces = []
            for _, test_data in enumerate(dataloader):
                logits = policy._model.forward_eval(test_data['obs'].permute(0, 3, 1, 2).float().cuda() / 255.)
                loss = criterion(logits, test_data['action'].cuda()).item()
                preds = torch.argmax(logits, dim=-1)
                acc = torch.sum((preds == test_data['action'].cuda())).item() / preds.shape[0]

                losses.append(loss)
                acces.append(acc)
            tb_logger.add_scalar('learn_iter/recurrent_train_loss', sum(losses) / len(losses), learner.train_iter)
            tb_logger.add_scalar('learn_iter/recurrent_train_acc', sum(acces) / len(acces), learner.train_iter)
    stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter)
    learner.call_hook('after_run')
    print('final reward is: {}'.format(reward))
    return policy, stop
