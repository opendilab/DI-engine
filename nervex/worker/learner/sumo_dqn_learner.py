"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. Alphastar implementation for supervised learning on linklink, including basic processes.
"""

import os.path as osp
import torch
import torch.nn as nn
import threading
import yaml
from easydict import EasyDict
from collections import OrderedDict

from nervex.worker.agent.sumo_dqn_agent import SumoDqnLearnerAgent
from nervex.data import build_dataloader
from nervex.optimizer.sumo_dqn_optimizer import SumoDqnOptimizer
from nervex.torch_utils import to_device
from nervex.utils import override, merge_dicts, pretty_print, read_config
from nervex.worker.learner.base_learner import Learner
from nervex.model.sumo_dqn.sumo_dqn_network import FCDQN
from nervex.envs.sumo.sumo_env import SumoWJ3Env
from nervex.data.collate_fn import sumo_dqn_collect_fn
from nervex.worker.agent.agent_plugin import TargetNetworkHelper

default_config = read_config(osp.join(osp.dirname(__file__), "alphastar_rl_learner_default_config.yaml"))


class SumoDqnLearner(Learner):
    _name = "SumoDqnLearner"

    def __init__(self, cfg, data_iterator):
        self.data_iterator = data_iterator

        plugin_cfg = OrderedDict({
            'grad': {
                'enable_grad': True
            },
        })
        if cfg.train.dqn.is_double:
            plugin_cfg['target_network'] = {'update_cfg': {'type': 'momentum', 'kwargs': {'theta': 0.99}}}
        self.plugin_cfg = OrderedDict(plugin_cfg)

        super(SumoDqnLearner, self).__init__(cfg)

        # Print and save config as metadata
        if self.rank == 0:
            pretty_print({"config": self.cfg})
            self.checkpoint_manager.save_config(self.cfg)

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
            collate_fn=sumo_dqn_collect_fn
        )
        return None, iter(dataloader), None

    @override(Learner)
    def _setup_agent(self):
        sumo_env = SumoWJ3Env({})
        model = FCDQN(sumo_env.info().obs_space.shape, [v for k, v in sumo_env.info().act_space.shape.items()])
        if self.use_cuda:
            model.cuda()
        agent = SumoDqnLearnerAgent(model, self.plugin_cfg)
        agent.mode(train=True)
        agent.target_mode(train=True)
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

    def _get_data_stat(self, data):
        """
        Overview: get the statistics of input data
        """
        return {}
