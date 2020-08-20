"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. Alphastar implementation for supervised learning on linklink, including basic processes.
"""

import os.path as osp

import torch
import torch.nn as nn
import threading

from nervex.worker.agent.sumo_dqn_agent import SumoDqnAgent
# from nervex.model import alphastar_model_default_config
from nervex.data import build_dataloader
from nervex.optimizer.sumo_dqn_optimizer import SumoDqnOptimizer
from nervex.torch_utils import to_device
from nervex.utils import override, merge_dicts, pretty_print, read_config
from nervex.worker.learner.base_learner import Learner
from nervex.data.fake_dataset import FakeSumoDataset

default_config = read_config(osp.join(osp.dirname(__file__), "alphastar_rl_learner_default_config.yaml"))


# def build_config(user_config):
#     """Aggregate a general config at the highest level class: Learner"""
#     default_config_with_model = merge_dicts(default_config, alphastar_model_default_config)
#     return merge_dicts(default_config_with_model, user_config)


class SumoDqnLearner(Learner):
    _name = "SumoDqnLearner"

    def __init__(self, cfg):
        # cfg = build_config(cfg)
        self.data_iterator = FakeSumoDataset(cfg.data.train.batch_size).getBatchSample()

        super(SumoDqnLearner, self).__init__(cfg)

        # Print and save config as metadata
        if self.rank == 0:
            pretty_print({"config": self.cfg})
            self.checkpoint_manager.save_config(self.cfg)

        # run thread
        run_thread = threading.Thread(target=self.run)
        run_thread.daemon = True
        run_thread.start()
      
    def run(self):
        super().run()
        super().finalize()

    @override(Learner)
    def _setup_data_source(self):
        dataloader = build_dataloader(
            self.data_iterator,
            self.cfg.data.train.dataloader_type,
            self.cfg.data.train.batch_size,
            self.use_distributed,
            read_data_fn=lambda x: x,
            collate_fn=lambda x: x
        )
        return None, iter(dataloader), None

    @override(Learner)
    def _setup_agent(self):
        #To add model
        model = FCDQN(380, [2, 2, 3])
        agent = SumoDqnAgent(model)
        agent.mode(train=True)
        return agent

    @override(Learner)
    def _setup_optimizer(self, model):
        # To change config, to know what cfg
        return SumoDqnOptimizer(self.agent, self.cfg.train)

    @override(Learner)
    def _preprocess_data(self, batch_data):
        data_stat = self._get_data_stat(batch_data)
        if self.use_cuda:
            batch_data = to_device(batch_data, 'cuda:{}'.format(self.rank % 8))
        return batch_data, data_stat

    @override(Learner)
    def _setup_stats(self):
        self.variable_record.register_var('cur_lr')
        self.variable_record.register_var('epoch')
        self.variable_record.register_var('data_time')
        self.variable_record.register_var('total_batch_time')

        self.tb_logger.register_var('cur_lr')
        self.tb_logger.register_var('epoch')
        self.tb_logger.register_var('total_batch_time')

        self.optimizer.register_stats(variable_record=self.variable_record, tb_logger=self.tb_logger)

    @override(Learner)
    def _setup_lr_scheduler(self, optimizer):
        torch_optimizer = optimizer.optimizer
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(torch_optimizer, milestones=[100000], gamma=1)
        return lr_scheduler

    @override(Learner)
    def _record_additional_info(self, iterations):
        pass

    @override(Learner)
    def evaluate(self):
        pass

    # @override(Learner)
    # def _update_data_priority(self, data, var_items):
    #     handle = data[0]  # get one frame of the trajectory
    #     # TODO(nyz) how to design priority for AS
    #     info = {
    #         'replay_unique_id': handle['replay_unique_id'],
    #         'replay_buffer_idx': handle['replay_buffer_idx'],
    #         'priority': [1.0 for _ in range(len(handle['replay_unique_id']))]
    #     }
    #     self.update_info(info)

    def _get_data_stat(self, data):
        """
        Overview: get the statistics of input data
        """
        return {}

    # @override(Learner)
    # def _get_model_state_dict(self):
    #     state_dict = self.agent.state_dict()
    #     state_dict = to_device(state_dict, 'cpu')
    #     if self.use_distributed:
    #         state_dict = {k[7:]: v for k, v in state_dict.items()}  # remove module.
    #     return {'state_dict': state_dict}

    # @override(Learner)
    # def _load_checkpoint_to_model(self, checkpoint):
    #     if self.use_distributed:
    #         prefix = 'module.'
    #         prefix_op = 'add'
    #     else:
    #         prefix = None
    #         prefix_op = None
    #     self.checkpoint_manager.load(
    #         checkpoint, self.agent.model, prefix=prefix, prefix_op=prefix_op, strict=False, need_torch_load=False
    #     )


class FCDQN(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim_list=[128, 256, 256], device='cpu'):
        super(FCDQN, self).__init__()
        self.act = nn.ReLU()
        layers = []
        for dim in hidden_dim_list:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(self.act)
            input_dim = dim
        self.main = nn.Sequential(*layers)
        self.action_dim = action_dim
        if isinstance(self.action_dim, list):
            self.pred = nn.ModuleList()
            for dim in self.action_dim:
                self.pred.append(nn.Linear(input_dim, dim))
        else:
            self.pred = nn.Linear(input_dim, action_dim)
        self.device = device

    def forward(self, x, info={}):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float)
        x = self.main(x)
        if isinstance(self.action_dim, list):
            x = [m(x) for m in self.pred]
        else:
            x = self.pred(x)
        return x
