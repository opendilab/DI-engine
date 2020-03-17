"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. Alphastar implementation for supervised learning on linklink, including basic processes.
"""

import torch

from pysc2.lib.static_data import ACTIONS_REORDER_INV, ACTIONS
from sc2learner.agent.alphastar_agent import AlphaStarAgent
from sc2learner.agent.model import build_model
from sc2learner.data import build_dataloader, build_dataset
from sc2learner.evaluate.supervised_criterion import SupervisedCriterion
from sc2learner.optimizer import AlphaStarSupervisedOptimizer
from sc2learner.torch_utils import to_device
from sc2learner.utils import override
from sc2learner.worker.learner.base_learner import SupervisedLearner


class AlphaStarSupervisedLearner(SupervisedLearner):
    _name = "AlphaStarSupervisedLearner"

    def __init__(self, cfg):
        self.data_stat = {
            'action_type': [k for k in ACTIONS],
            'delay': ['0-5', '6-22', '23-44', '44-64'],
            'queued': ['no_attr', 'no_queued', 'queued'],
            'selected_units': ['no_attr', '1', '2-8', '9-32', '33-64', '64+'],
            'target_units': ['no_attr', 'target_units'],
            'target_location': ['no_attr', 'target_location'],
        }
        super(AlphaStarSupervisedLearner, self).__init__(cfg)
        self.eval_criterion = SupervisedCriterion()

    @override(SupervisedLearner)
    def _setup_data_source(self):
        cfg = self.cfg
        dataset = build_dataset(cfg.data.train)
        eval_dataset = build_dataset(cfg.data.eval)
        dataloader = build_dataloader(cfg.data.train, dataset)
        eval_dataloader = build_dataloader(cfg.data.eval, eval_dataset)
        return dataset, dataloader, eval_dataloader

    @override(SupervisedLearner)
    def _setup_agent(self):
        agent = AlphaStarAgent(self.cfg, build_model, self.use_cuda, self.use_distributed)
        agent.train()
        return agent

    @override(SupervisedLearner)
    def _setup_optimizer(self, model):
        return AlphaStarSupervisedOptimizer(self.cfg, self.agent, self.use_distributed, self.world_size)

    @override(SupervisedLearner)
    def _preprocess_data(self, batch_data):
        data_stat = self._get_data_stat(batch_data)
        if self.use_cuda:
            batch_data = to_device(batch_data, 'cuda')
        return batch_data, data_stat

    @override(SupervisedLearner)
    def _setup_stats(self):

        self.variable_record.register_var('cur_lr')
        self.variable_record.register_var('epoch')
        self.variable_record.register_var('data_time')
        self.variable_record.register_var('total_batch_time')
        self.tb_logger.register_var('cur_lr')
        self.tb_logger.register_var('epoch')
        self.tb_logger.register_var('total_batch_time')

        self.optimizer.register_stats(variable_record=self.variable_record, tb_logger=self.tb_logger)

        self.variable_record.register_var('action_type', var_type='1darray',
                                          var_item_keys=self.data_stat['action_type'])  # noqa
        self.tb_logger.register_var('action_type', var_type='histogram')

        for k in (set(self.data_stat.keys()) - {'action_type'}):
            self.variable_record.register_var(k, var_type='1darray', var_item_keys=self.data_stat[k])
            self.tb_logger.register_var(k, var_type='scalars')

    @override(SupervisedLearner)
    def _setup_lr_scheduler(self, optimizer):
        torch_optimizer = optimizer.optimizer
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(torch_optimizer, milestones=[100000], gamma=1)
        return lr_scheduler

    @override(SupervisedLearner)
    def _record_additional_info(self, iterations):
        histogram_keys = ['action_type']
        scalars_keys = self.data_stat.keys() - histogram_keys
        self.tb_logger.add_val_list(self.variable_record.get_vars_tb_format(
            scalars_keys, iterations, var_type='1darray', viz_type='scalars'), viz_type='scalars')
        self.tb_logger.add_val_list(self.variable_record.get_vars_tb_format(
            histogram_keys, iterations, var_type='1darray', viz_type='histogram'), viz_type='histogram')

    @override(SupervisedLearner)
    def evaluate(self):
        self.agent.eval()
        for data_index, data in enumerate(self.eval_dataloader):
            self.agent.reset_previous_state(data[0]["start_step"])
            for step, step_data in enumerate(data):
                if self.use_cuda:
                    step_data = to_device(step_data, "cuda")
                actions, _, _ = self.agent.compute_action(step_data, mode="evaluate")
                self.eval_criterion.update(actions, step_data['actions'])
                if step % 100 == 0:
                    args = [self.rank, data_index + 1, len(self.eval_dataloader), step, len(data)]
                    self.logger.info('EVAL[rank: {}](sample: {}/{})(step: {}/{})'.format(*args))
        eval_result = self.eval_criterion.get_stat()
        self.logger.info(eval_result)
        self.agent.train()

    def _get_data_stat(self, data):
        """
            Overview: empty interface for data statistics
            Arguments:
                - data (:obj:`dict`): data dict for one step iteration
            Returns:
                - (:obj`dict`): data statistics(default empty dict)
        """
        data_stat = {k: {t: 0 for t in v} for k, v in self.data_stat.items()}
        for step_data in data:
            action = step_data['actions']
            for k, v in action.items():
                if k == 'action_type':
                    for t in v:
                        data_stat[k][ACTIONS_REORDER_INV[t.item()]] += 1
                elif k == 'delay':
                    for t in v:
                        if t <= 5:
                            data_stat[k]['0-5'] += 1
                        elif t <= 22:
                            data_stat[k]['6-22'] += 1
                        elif t <= 44:
                            data_stat[k]['23-44'] += 1
                        elif t <= 64:
                            data_stat[k]['44-64'] += 1
                        else:
                            raise ValueError("invalid delay value: {}".format(t))
                elif k == 'queued':
                    for t in v:
                        if not isinstance(t, torch.Tensor):
                            data_stat[k]['no_attr'] += 1
                        elif t == 0:
                            data_stat[k]['no_queued'] += 1
                        elif t == 1:
                            data_stat[k]['queued'] += 1
                        else:
                            raise ValueError("invalid queued value: {}".format(t))
                elif k == 'selected_units':
                    for t in v:
                        if not isinstance(t, torch.Tensor):
                            data_stat[k]['no_attr'] += 1
                        else:
                            num = t.shape[0]
                            if num <= 0:
                                raise ValueError("invalid queued value: {}".format(t))
                            elif num <= 1:
                                data_stat[k]['1'] += 1
                            elif num <= 8:
                                data_stat[k]['2-8'] += 1
                            elif num <= 32:
                                data_stat[k]['9-32'] += 1
                            elif num <= 64:
                                data_stat[k]['33-64'] += 1
                            else:
                                data_stat[k]['64+'] += 1
                elif k == 'target_units':
                    for t in v:
                        if not isinstance(t, torch.Tensor):
                            data_stat[k]['no_attr'] += 1
                        else:
                            data_stat[k]['target_units'] += 1
                elif k == 'target_location':
                    for t in v:
                        if not isinstance(t, torch.Tensor):
                            data_stat[k]['no_attr'] += 1
                        else:
                            data_stat[k]['target_location'] += 1
        data_stat = {k: list(v.values()) for k, v in data_stat.items()}
        return data_stat
