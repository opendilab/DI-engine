"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. Alphastar implementation for supervised learning on linklink, including basic processes.
"""

import os.path as osp

import torch
import threading

from sc2learner.worker.agent.alphastar_agent import AlphaStarAgent
from sc2learner.model import alphastar_model_default_config
from sc2learner.data import build_dataloader
from sc2learner.optimizer import AlphaStarRLOptimizer
from sc2learner.torch_utils import to_device
from sc2learner.utils import override, merge_dicts, pretty_print, read_config
from sc2learner.worker.learner.base_rl_learner import BaseRLLearner

default_config = read_config(osp.join(osp.dirname(__file__), "alphastar_rl_learner_default_config.yaml"))


def build_config(user_config):
    """Aggregate a general config at the highest level class: Learner"""
    default_config_with_model = merge_dicts(default_config, alphastar_model_default_config)
    return merge_dicts(default_config_with_model, user_config)


class AlphaStarRLLearner(BaseRLLearner):
    _name = "AlphaStarRLLearner"

    def __init__(self, cfg):
        cfg = build_config(cfg)
        super(AlphaStarRLLearner, self).__init__(cfg)

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

    @override(BaseRLLearner)
    def _setup_data_source(self):
        dataloader = build_dataloader(
            self.data_iterator,
            self.cfg.data.train.dataloader_type,
            self.cfg.data.train.batch_size,
            self.use_distributed,
            read_data_fn=self.load_trajectory,
        )
        return None, iter(dataloader), None

    @override(BaseRLLearner)
    def _setup_agent(self):
        agent = AlphaStarAgent(self.cfg.model, self.cfg.data.train.batch_size, self.use_cuda, self.use_distributed)
        agent.train()
        return agent

    @override(BaseRLLearner)
    def _setup_optimizer(self, model):
        return AlphaStarRLOptimizer(self.agent, self.cfg)

    @override(BaseRLLearner)
    def _preprocess_data(self, batch_data):
        data_stat = self._get_data_stat(batch_data)
        if self.use_cuda:
            batch_data = to_device(batch_data, 'cuda:{}'.format(self.rank % 8))
        return batch_data, data_stat

    @override(BaseRLLearner)
    def _setup_stats(self):
        self.variable_record.register_var('cur_lr')
        self.variable_record.register_var('epoch')
        self.variable_record.register_var('data_time')
        self.variable_record.register_var('total_batch_time')

        self.tb_logger.register_var('cur_lr')
        self.tb_logger.register_var('epoch')
        self.tb_logger.register_var('total_batch_time')

        self.optimizer.register_stats(variable_record=self.variable_record, tb_logger=self.tb_logger)

    @override(BaseRLLearner)
    def _setup_lr_scheduler(self, optimizer):
        torch_optimizer = optimizer.optimizer
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(torch_optimizer, milestones=[100000], gamma=1)
        return lr_scheduler

    @override(BaseRLLearner)
    def _record_additional_info(self, iterations):
        pass

    @override(BaseRLLearner)
    def evaluate(self):
        pass

    @override(BaseRLLearner)
    def _update_data_priority(self, data, var_items):
        handle = data[0]  # get one frame of the trajectory
        # TODO(nyz) how to design priority for AS
        info = {
            'replay_unique_id': handle['replay_unique_id'],
            'replay_buffer_idx': handle['replay_buffer_idx'],
            'priority': [1.0 for _ in range(len(handle['replay_unique_id']))]
        }
        self.update_info(info)

    def _get_data_stat(self, data):
        """
        Overview: get the statistics of input data
        """
        return {}

    @override(BaseRLLearner)
    def _get_model_state_dict(self):
        state_dict = self.agent.get_model().state_dict()
        state_dict = to_device(state_dict, 'cpu')
        if self.use_distributed:
            state_dict = {k[7:]: v for k, v in state_dict.items()}  # remove module.
        return {'state_dict': state_dict}

    @override(BaseRLLearner)
    def _load_checkpoint_to_model(self, checkpoint):
        if self.use_distributed:
            prefix = 'module.'
            prefix_op = 'add'
        else:
            prefix = None
            prefix_op = None
        self.checkpoint_manager.load(
            checkpoint, self.agent.get_model(), prefix=prefix, prefix_op=prefix_op, strict=False, need_torch_load=False
        )
