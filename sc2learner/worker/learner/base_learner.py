"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. base class for supervised learning on linklink, including basic processes.
"""
import numbers
import os

import numpy as np
import torch

from sc2learner.worker.agent.alphastar_agent import BaseAgent
from sc2learner.torch_utils import build_checkpoint_helper, auto_checkpoint, CountVar
from sc2learner.utils import build_logger, dist_init, dist_finalize, allreduce, EasyTimer


class Learner:
    """
        Overview: base class for supervised learning on linklink, including basic processes.
        Interface: __init__, run, finalize, save_checkpoint, eval
    """
    _name = "BaseSupervisedLearner"  # override this variable for high-level learner

    def __init__(self, cfg):
        """
            Overview: initialization method, using setting to build model, dataset, optimizer, lr_scheduler
                      and other helper. It can also load checkpoint.
            Arguments:
                - cfg (:obj:`dict`): learner config

             # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # for debug async CUDA
        """
        assert "Base" not in self._name, "You should subclass base learner to get a runnable learner!"

        # parse config
        self.cfg = cfg
        self.use_distributed = cfg.train.use_distributed
        self.use_cuda = cfg.train.use_cuda
        self.train_dataloader_type = cfg.data.train.dataloader_type
        if self.train_dataloader_type == 'epoch':
            self.max_epochs = cfg.train.max_epochs
            self.max_iterations = np.inf
        else:
            self.max_iterations = int(float(cfg.train.max_iterations))
        self.use_cuda = cfg.train.use_cuda
        if self.use_distributed:
            self.rank, self.world_size = dist_init()  # initialize rank and world size for linklink
        else:
            self.rank, self.world_size = 0, 1
        if self.use_cuda and not torch.cuda.is_available():
            import logging
            logging.error("You do not have GPU! If you are not testing locally, something is going wrong.")
            self.use_cuda = False
        assert self.train_dataloader_type in ['epoch', 'iter', 'online']

        # build model
        self.agent = self._setup_agent()  # build model by policy from alphaStar
        assert isinstance(self.agent, BaseAgent)

        # build data source
        self.dataset, self.dataloader, self.eval_dataloader = self._setup_data_source()

        # build optimizer
        self.optimizer = self._setup_optimizer(self.agent)
        self.lr_scheduler = self._setup_lr_scheduler(self.optimizer)

        # build logger
        if self.rank == 0:  # only one thread need to build logger
            self.logger, self.tb_logger, self.variable_record = self._setup_logger(self.rank)
            #self.logger.info('cfg:\n{}'.format(self.cfg))
            #self.logger.info('model:\n{}'.format(self.agent))
            self._setup_stats()
        else:
            self.logger, _, _ = self._setup_logger(self.rank)
        self.last_iter = CountVar(init_val=0)  # count for iterations
        self.last_epoch = CountVar(init_val=0)  # count for epochs
        self.data_timer = EasyTimer()
        self.total_timer = EasyTimer()

        # build checkpoint helper
        self._setup_checkpoint_manager()

    def _setup_data_source(self):
        raise NotImplementedError()

    def _setup_agent(self):
        """Build the agent object of learner"""
        raise NotImplementedError()

    def _setup_optimizer(self, model):
        """Build a sensestar optimizer"""
        raise NotImplementedError()

    def evaluate(self):
        pass

    def _setup_stats(self):
        """Setup algorithm specify statistics."""
        pass

    def _setup_lr_scheduler(self, optimizer):
        """Build lr scheduler"""
        return None

    def _record_additional_info(self, iterations):
        """
            Overview: empty interface to record additional info on logger
            Arguments:
                - iterations (:obj:`int`): iteration number
        """
        pass

    def _update_data_priority(self, data, var_items):
        pass

    def _preprocess_data(self, data):
        return data

    # === Functions that should not be override. ===
    def _setup_logger(self, rank):
        """Setup logger"""
        return build_logger(self.cfg, rank=rank)

    def _setup_checkpoint_manager(self):
        self.checkpoint_manager = build_checkpoint_helper(self.cfg.common.save_path, self.rank)
        if self.train_dataloader_type == 'epoch':
            self.ckpt_dataset = self.dataset
        elif self.train_dataloader_type in ['iter', 'online']:
            # iter type doesn't save some context
            self.ckpt_dataset = None
        else:
            raise NotImplementedError()
        if self.cfg.common.load_path != '':
            self.restore(self.cfg.common.load_path)
            self.last_epoch.add(1)  # skip interrupted epoch
            self.last_iter.add(1)  # skip interrupted iter

    def _check_checkpoint_path(self, ckpt):
        """ Validate the checkpoint """
        return os.path.exists(ckpt)

    def restore(self, checkpoint_path=None):
        checkpoint_path = checkpoint_path or self.cfg.common.load_path
        ckpt_ok = self._check_checkpoint_path(checkpoint_path)
        if ckpt_ok:
            self.checkpoint_manager.load(
                checkpoint_path,
                self.agent.get_model(),
                optimizer=self.optimizer,
                last_iter=self.last_iter,
                last_epoch=self.last_epoch,  # TODO last_epoch for lr_scheduler
                dataset=self.ckpt_dataset,
                logger_prefix='({})'.format(self._name)
            )

    def save_checkpoint(self):
        """
            Overview: save checkpoint named by current iteration(only rank 0)
        """
        if self.rank == 0:
            self.checkpoint_manager.save_iterations(
                self.last_iter.val,
                self.agent.get_model(),
                optimizer=self.optimizer,
                dataset=self.dataset,
                last_epoch=self.last_epoch.val
            )

    @auto_checkpoint
    def run(self):
        """
            Overview: train / evaluate model with dataset in numbers of epoch. Main loop of Learner.
        """

        if self.cfg.common.only_evaluate:
            self.evaluate()
            return

        if self.train_dataloader_type == 'epoch':
            while self.last_epoch.val < self.max_epochs:
                # TODO(pzh) need further consideration on this function.
                if hasattr(self.dataloader.dataset, 'step'):  # call dataset.step()
                    self.dataloader.dataset.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()  # update lr
                self._run()
                self.last_epoch.add(1)
        elif self.train_dataloader_type in ['iter', 'online']:
            self._run()

        self.save_checkpoint()

    def _run(self):
        while self.last_iter.val < self.max_iterations:
            with self.total_timer:
                with self.data_timer:
                    try:
                        batch_data = next(iter(self.dataloader))
                    except StopIteration:  # for limited length dataloader
                        return
                    processed_data, data_stats = self._preprocess_data(batch_data)
                var_items, time_stats = self.optimizer.learn(processed_data)
                var_items['cur_lr'] = self.lr_scheduler.get_lr()[0]
                var_items['epoch'] = self.last_epoch.val
                var_items.update(data_stats)

                self._update_data_priority(processed_data, var_items)
            time_stats.update(
                data_time=self.data_timer.value,
                total_batch_time=self.total_timer.value,
            )
            self._manage_learning_information(var_items, time_stats)

            self.last_iter.add(1)
            if self.last_iter.val % self.cfg.logger.eval_freq == 0:
                self.evaluate()

    def _manage_learning_information(self, var_items, time_items):

        if self.use_distributed:
            var_items = aggregate(var_items)
            time_items = aggregate(time_items)

        if self.rank != 0:
            return

        keys = list(self.variable_record.get_var_names('scalar')) + list(self.variable_record.get_var_names('1darray'))
        self.variable_record.update_var(transform_dict(var_items, keys))
        self.variable_record.update_var(time_items)

        iterations = self.last_iter.val

        if iterations % self.cfg.logger.print_freq == 0:
            self.logger.info("=== Training Iteration {} Result ===".format(self.last_iter.val))
            self.logger.info('iterations:{}\t{}'.format(iterations, self.variable_record.get_vars_text()))
            tb_keys = self.tb_logger.scalar_var_names
            self.tb_logger.add_val_list(
                self.variable_record.get_vars_tb_format(tb_keys, iterations, var_type='scalar'), viz_type='scalar'
            )
            self._record_additional_info(iterations)

        if iterations % self.cfg.logger.save_freq == 0:
            self.save_checkpoint()

    def finalize(self):
        """ Overview: finalize, called after training """
        if self.use_distributed:
            dist_finalize()


class SupervisedLearner(Learner):
    """An abstract supervised learning learner class"""
    _name = "BaseSupervisedLearner"


def transform_dict(var_items, keys):
    new_dict = {}
    for k in keys:
        if k in var_items.keys():
            v = var_items[k]
            if isinstance(v, torch.Tensor):
                if v.shape == (1, ):
                    v = v.item()  # get item
                else:
                    v = v.tolist()
            else:
                v = v
            new_dict[k] = v
    return new_dict


def aggregate(data):
    """
        Overview: merge all info from other rank
        Arguments:
            - data (:obj:`dict`): data needs to be reduced. Could be dict, torch.Tensor,
                                  numbers.Integral or numbers.Real
        Returns:
            - (:obj`dict`): data after reduce
    """
    if isinstance(data, dict):
        new_data = {}
        for k, v in data.items():
            new_data[k] = aggregate(v)
    elif isinstance(data, list):
        new_data = []
        for t in data:
            new_data.append(aggregate(t))
    elif isinstance(data, torch.Tensor):
        new_data = data.clone()
        allreduce(new_data)  # get data from other processes
    elif isinstance(data, numbers.Integral) or isinstance(data, numbers.Real):
        new_data = torch.scalar_tensor(data).reshape([1])
        allreduce(new_data)
        new_data = new_data.item()
    else:
        raise TypeError("invalid info type: {}".format(type(data)))
    return new_data
