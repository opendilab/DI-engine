from typing import Union, Optional, List, Any, Tuple
import time
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from ding.worker import BaseLearner, LearnerHook, MetricSerialEvaluator, IMetric
from ding.config import read_config, compile_config
from ding.model.template.q_learning import DQN
from ding.utils import set_pkg_seed, get_rank, dist_init
from ding.policy import BehaviourCloningPolicy
from dizoo.classic_control.cartpole.config.cartpole_bc_config import main_config, create_config
from ding.worker import create_serial_collector
from ding.envs import get_vec_env_setting, create_env_manager
from functools import partial
import numpy as np
from ding.policy.common_utils import default_preprocess_learn
import pickle
class BCMetric(IMetric):

    def __init__(self) -> None:
        self.loss = torch.nn.CrossEntropyLoss()

    @staticmethod
    def accuracy(inputs: torch.Tensor, label: torch.Tensor) -> dict:
        """Computes the accuracy over the k top predictions for the specified values of k"""
        batch_size = label.size(0)
        _, pred = inputs.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.reshape(1, -1).expand_as(pred))
        return {'acc': correct.reshape(-1).float().sum(0) * 100. / batch_size}
        
    def eval(self, inputs: dict, label: torch.Tensor) -> dict:
        """
        Returns:
            - eval_result (:obj:`dict`): {'loss': xxx, 'acc': xxx}
        """
        inputs=inputs['logit']
        loss = self.loss(inputs, label)
        output = self.accuracy(inputs, label)
        output['loss'] = loss
        for k in output:
            output[k] = output[k].item()
        return output

    def reduce_mean(self, inputs: List[dict]) -> dict:
        L = len(inputs)
        output = {}
        for k in inputs[0].keys():
            output[k] = sum([t[k] for t in inputs]) / L
        return output

    def gt(self, metric1: dict, metric2: dict) -> bool:
        if metric2 is None:
            return True
        if 'loss' in metric1:
            for k in metric1:
                if k != 'loss':
                    if metric1[k] < metric2[k] or metric1['loss'] > metric2['loss']:
                        return False

        else:
            for k in metric1:
                if metric1[k] < metric2[k]:
                    return False
        return True

class CustomDataset(Dataset):
    def __init__(self, text, labels):
        self.labels = labels
        self.text = text
    def __len__(self):
            return len(self.labels)
    def __getitem__(self, idx):
            label = self.labels[idx]
            text = self.text[idx]
            sample=[text,label]#sample = {"obs": text, "action": label}
            return sample

def data_process(data):
    data = default_preprocess_learn(data)
    newlist = list()
    for key in data.keys():
        newlist.append(key)
    for key in newlist:
        if key not in ['obs', 'action']:
            del data[key]
    return CustomDataset(data['obs'], data['action'])


def main(cfg: dict, create_cfg: dict, seed: int, env_setting: Optional[List[Any]] = None) -> None:
    cfg = compile_config(cfg, seed=seed, policy=BehaviourCloningPolicy, auto=True, create_cfg=create_cfg, evaluator=MetricSerialEvaluator)
    if cfg.policy.learn.multi_gpu:
        rank, world_size = dist_init()
    else:
        rank, world_size = 0, 1

    # Random seed
    set_pkg_seed(cfg.seed + rank, use_cuda=cfg.policy.cuda)
    
    # Prepare data
    model = DQN(obs_shape = cfg.policy.model.obs_shape, action_shape = cfg.policy.model.action_shape, encoder_hidden_size_list = cfg.policy.model.encoder_hidden_size_list)
    if cfg.policy.collect.demonstration_model_path is None:
        policy = BehaviourCloningPolicy(cfg.policy, model=model, enable_field=['learn', 'eval'])
        with open('/Users/nieyunpeng/Documents/open-sourced-algorithms/BC_DI-engine/dizoo/behaviour_cloning/entry/expert_data.pkl', 'rb') as f:
            data = pickle.load(f)
            length = len(data)
            train_data_length = length * 9 // 10
            text_data_length = length - train_data_length

        learn_dataset = data[:train_data_length]
        eval_dataset = data[train_data_length:]

        if cfg.policy.learn.multi_gpu:
            learn_sampler = DistributedSampler(learn_dataset)
            eval_sampler = DistributedSampler(eval_dataset)
        else:
            learn_sampler, eval_sampler = None, None
        learn_dataset = data_process(learn_dataset)
        eval_dataset = data_process(eval_dataset)
        #learn_dataloader = DataLoader(learn_dataset, cfg.policy.learn.batch_size, shuffle=True)
        #eval_dataloader = DataLoader(eval_dataset, cfg.policy.eval.batch_size, shuffle=True)
        learn_dataloader = DataLoader(learn_dataset, cfg.policy.learn.batch_size, sampler=learn_sampler, num_workers=3)
        eval_dataloader = DataLoader(eval_dataset, cfg.policy.eval.batch_size, sampler=eval_sampler, num_workers=2)
    else:
        policy = BehaviourCloningPolicy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])
        if env_setting is None:
            env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
        else:
            env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
        collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
        policy.collect_mode.load_state_dict(torch.load(cfg.policy.collect.demonstration_model_path, map_location = 'cpu'))
        collector = create_serial_collector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode,
        exp_name=cfg.exp_name
    )

        learn_dataset = collector.collect(n_sample=10000)
        eval_dataset = collector.collect(n_sample=10000)
        learn_dataset = data_process(learn_dataset)
        eval_dataset = data_process(eval_dataset)
        if cfg.policy.learn.multi_gpu:
            learn_sampler = DistributedSampler(learn_dataset)
            eval_sampler = DistributedSampler(eval_dataset)
        else:
            learn_sampler, eval_sampler = None, None
        learn_dataloader = DataLoader(learn_dataset, cfg.policy.learn.batch_size, sampler=learn_sampler, num_workers=3)
        eval_dataloader = DataLoader(eval_dataset, cfg.policy.eval.batch_size, sampler=eval_sampler, num_workers=2)


    # Main components
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    eval_metric = BCMetric()
    evaluator = MetricSerialEvaluator(
        cfg.policy.eval.evaluator, [eval_dataloader, eval_metric], policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    # ==========
    # Main loop
    # ==========
    learner.call_hook('before_run')
    end = time.time()

    for epoch in range(cfg.policy.learn.train_epoch):
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, epoch, 0)
            if stop:
                break
        for i, train_data in enumerate(learn_dataloader):
            learner.data_time = time.time() - end
            learner.epoch_info = (epoch, i, len(learn_dataloader))
            learner.train(train_data)
            end = time.time()
        learner.policy.get_attribute('lr_scheduler').step()

    learner.call_hook('after_run')


if __name__ == "__main__":
    main(main_config, create_config, 0)
