from typing import Union, Optional, List, Any, Tuple
import time
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from ding.worker import BaseLearner, LearnerHook, MetricSerialEvaluator, IMetric, create_serial_collector, InteractionSerialEvaluator, BaseSerialCommander, create_buffer
from ding.config import read_config, compile_config
from ding.model.template.q_learning import DQN
from ding.utils import set_pkg_seed, get_rank, dist_init
from ding.policy import BehaviourCloningObservationPolicy
from dizoo.classic_control.cartpole.config.cartpole_bc_config import main_config, create_config
from ding.envs import get_vec_env_setting, create_env_manager
from functools import partial
import numpy as np
from ding.policy.common_utils import default_preprocess_learn
import pickle
import torch.nn as nn
from ding.entry.application_entry import collect_demo_data
from ding.policy import create_policy, PolicyFactory

class CustomDataset(Dataset):
    def __init__(self, text, labels):
        self.labels = labels
        self.text = text
    def __len__(self):
            return len(self.labels)
    def __getitem__(self, idx):
            label = self.labels[idx]
            text = self.text[idx]
            sample = {"obs": text, "action": label}
            return sample

def data_process_BCO_agentdata(data):
    data = default_preprocess_learn(data)
    newlist = list()
    for key in data.keys():
        newlist.append(key)
    for key in newlist:
        if key not in ['obs', 'next_obs', 'action']:
            del data[key]
    return CustomDataset(torch.cat((data['obs'], data['next_obs']),1), data['action'])

#CustomDataset(torch.argmax(m(model(torch.cat(data['obs'], data['next_obs']))),-1), data['action'])
def data_process_BCO_expertdata(data):
    data = default_preprocess_learn(data)
    newlist = list()
    for key in data.keys():
        newlist.append(key)
    for key in newlist:
        if key not in ['obs', 'next_obs']:
            del data[key]
    return torch.cat((data['obs'], data['next_obs']),1)

def train_state_trainsition_model(training_set, model, n_epoch):
    '''
    train transition model, given pair of states return action (s0,s1 ---> a0 if n=2)
    Input:
    training_set: 
    model: transition model want to train
    n: window size (how many states needed to predict the next action)
    batch_size: batch size
    n_epoch: number of epoches
    return:
    model: trained transition model
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    loss_list = []
    for itr in range(n_epoch):
        total_loss = 0
        data   = training_set['obs']
        y      = training_set['action']
        y_pred = model(data)
        loss   = criterion(y_pred, y)
        '''
        print('y')
        print(y)
        print('pred y')
        print(y_pred)
        '''
        total_loss += loss.item() 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("[EPOCH]: %i, [LOSS]: %.6f" % (itr+1, total_loss))
        loss_list.append(total_loss / training_set['obs'].shape[0])
    return model

def main(cfg: dict, create_cfg: dict, seed: int, env_setting: Optional[List[Any]] = None) -> None:

    create_cfg.policy.type = create_cfg.policy.type + '_command'
    cfg = compile_config(cfg, seed=seed, policy=BehaviourCloningObservationPolicy, auto=True, create_cfg=create_cfg, save_cfg=True)
    if cfg.policy.learn.multi_gpu:
        rank, world_size = dist_init()
    else:
        rank, world_size = 0, 1

    state_trainsition_model = nn.Sequential(
        
        nn.Linear(cfg.policy.model.obs_shape * 2, 60),
        nn.ReLU(),
        
        nn.Linear(60, 80),
        nn.ReLU(),
        
        nn.Linear(80, 100),
        nn.ReLU(),
        
        nn.Linear(100, 70),
        nn.ReLU(),
        
        nn.Linear(70, 40),
        nn.ReLU(),
        
        nn.Linear(40, 10),
        nn.ReLU(),
        
        nn.Linear(10, cfg.policy.model.action_shape))
    # Random seed
    set_pkg_seed(cfg.seed + rank, use_cuda=cfg.policy.cuda)

    # Generate Expert Data 
    policy_model = DQN(obs_shape = cfg.policy.model.obs_shape, action_shape = cfg.policy.model.action_shape)
    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    if cfg.policy.collect.demonstration_model_path is None:
        policy = BehaviourCloningObservationPolicy(cfg.policy, model=policy_model, enable_field=['learn', 'collect', 'eval'])
        with open(cfg.policy.collect.demonstration_offline_data_path, 'rb') as f:
            data = pickle.load(f)
            length = len(data)
            train_data_length = length * 9 // 10
            text_data_length = length - train_data_length

        learn_dataset = data[:train_data_length]
        eval_dataset = data[train_data_length:]
        expert_learn_dataset = data_process_BCO_expertdata(learn_dataset)
        expert_eval_dataset = data_process_BCO_expertdata(eval_dataset)
        #expert_learn_dataloader = DataLoader(learn_dataset, cfg.policy.learn.batch_size, shuffle=True)
        #expert_eval_dataloader = DataLoader(eval_dataset, cfg.policy.eval.batch_size, shuffle=True)
        #expert_learn_dataloader = DataLoader(learn_dataset, cfg.policy.learn.batch_size, sampler=learn_sampler, num_workers=3)
        #eval_dataloader = DataLoader(eval_dataset, cfg.policy.eval.batch_size, sampler=eval_sampler, num_workers=2)
    else:
        policy = BehaviourCloningObservationPolicy(cfg.policy, model=policy_model, enable_field=['learn', 'collect', 'eval'])
        collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
        '''
        action_space = collector_env.env_info().act_space
        random_policy = PolicyFactory.get_random_policy(policy.collect_mode, action_space=action_space)
        collector.reset_policy(random_policy)
        collect_kwargs = commander.step()
        new_data = collector.collect(n_sample=cfg.policy.random_collect_size, policy_kwargs=collect_kwargs)
        replay_buffer.push(new_data, cur_collector_envstep=0)
        collector.reset_policy(policy.collect_mode)
        '''

        
        policy.collect_mode.load_state_dict(torch.load(cfg.policy.collect.demonstration_model_path, map_location = 'cpu'))
        collector = create_serial_collector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode,
        exp_name=cfg.exp_name
    )

        learn_dataset = collector.collect(n_sample=100000)
        eval_dataset = collector.collect(n_sample=100000)
        expert_learn_dataset = data_process_BCO_expertdata(learn_dataset)
        expert_eval_dataset = data_process_BCO_expertdata(eval_dataset)
        #expert_learn_dataloader = DataLoader(learn_dataset, cfg.policy.learn.batch_size, sampler=learn_sampler, num_workers=3)
        #eval_dataloader = DataLoader(eval_dataset, cfg.policy.eval.batch_size, sampler=eval_sampler, num_workers=2)
        collector.reset_policy(policy.collect_mode)   #check??

    # Main components
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    #log_show_hook = ImageClsLogShowHook(
    #    name='image_cls_log_show_hook', priority=0, position='after_iter', freq=cfg.policy.learn.learner.log_show_freq
    #)
    #learner.register_hook(log_show_hook)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
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
    # ==========
    # Main loop
    # ==========
    #replay_buffer = create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)  #??
    #commander = BaseSerialCommander(
    #    cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode
    #)
    learner.call_hook('before_run')
    end = time.time()
    learned_model = None
    m = nn.Softmax(dim=-1)
    for epoch in range(cfg.policy.learn.train_epoch):
        #collect_kwargs = commander.step()
        if learned_model is None:
            learned_model = state_trainsition_model
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, epoch, 0)
            if stop:
                break
        new_data = collector.collect(train_iter=learner.train_iter)
        learn_dataset = data_process_BCO_agentdata(new_data)
        learn_dataloader = DataLoader(learn_dataset, cfg.policy.learn.batch_size, sampler=None, num_workers=3)
        for i, train_data in enumerate(learn_dataloader):
            learned_model = train_state_trainsition_model(train_data, learned_model, 10)
        expert_learn_dataset_obj = CustomDataset(expert_learn_dataset[:, 0:int(expert_learn_dataset.shape[1]/2)], torch.argmax(m(learned_model(expert_learn_dataset)),-1))
        expert_learn_dataloader = DataLoader(expert_learn_dataset_obj, cfg.policy.learn.batch_size, sampler=None, num_workers=3)  
        for i, train_data in enumerate(expert_learn_dataloader):
            learner.data_time = time.time() - end
            learner.epoch_info = (epoch, i, len(learn_dataloader))
            learner.train(train_data)
            end = time.time()
        learner.policy.get_attribute('lr_scheduler').step()

    learner.call_hook('after_run')


if __name__ == "__main__":
    main(main_config, create_config, 0)
