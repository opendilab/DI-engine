"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. base class for model learning(SL/RL) on linklink, including basic processes.
"""
import numbers
import os

import numpy as np
import torch

from nervex.worker.agent.alphastar_agent import BaseAgent
from nervex.torch_utils import build_checkpoint_helper, auto_checkpoint, CountVar
from nervex.utils import build_logger, dist_init, dist_finalize, allreduce, EasyTimer


class Learner:
    r"""
    Overview:
        base class for model learning(SL/RL), which uses linklink for multi-GPU learning
    Interface:
        __init__, run, finalize, save_checkpoint, evaluate, restore
    """
    _name = "BaseSupervisedLearner"  # override this variable for high-level learner

    def __init__(self, cfg):
        """
        Overview:
            initialization method, using config setting to build model, dataset, optimizer, lr_scheduler
            and other helper. It can also load and save checkpoint.
        Arguments:
            - cfg (:obj:`dict`): learner config, you can view `learner_cfg <../../../configuration/index.html>`_\
            for reference
        Notes:
            if you want to debug in sync CUDA mode, please use the following line code in the beginning of `__init__`.

            os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # for debug async CUDA
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
        self.last_frame = CountVar(init_val=0)  # count for frames
        self.data_timer = EasyTimer()
        self.total_timer = EasyTimer()

        # build checkpoint helper
        self._setup_checkpoint_manager()

    def _setup_data_source(self):
        raise NotImplementedError()

    def _setup_agent(self):
        """Build the agent object of learner, which is the runtime object of model"""
        raise NotImplementedError()

    def _setup_optimizer(self, model):
        """Build a training optimizer"""
        raise NotImplementedError()

    def evaluate(self):
        r"""
        Overview:
            evaluate training result(usually used in SL/IL setting)
        """
        pass

    def _setup_stats(self):
        """Setup algorithm specify statistics."""
        pass

    def _setup_lr_scheduler(self, optimizer):
        """
        Overview:
            setup lr scheduler, you can refer to `PyTorch lr_scheduler interface <https://pytorch.org/docs/master/\
            optim.html#how-to-adjust-learning-rate>`_ for reference, we also implement some customized lr_schedulers
        Arguments:
            - optimizer (:obj:`torch.optim.Optimizer`): optimizer
        """
        pass

    def _record_additional_info(self, iterations):
        r"""
        Overview:
            empty interface to record additional info on logger, learner subclass can override this inferface to
            add its own information into logger and tensorboard.
        Arguments:
            - iterations (:obj:`int`): iteration number
        """
        pass

    def _update_data_priority(self, data, var_items):
        pass

    def _preprocess_data(self, data):
        """
        Overview:
            interface for specific preprocess for input data
        """
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
            self.last_frame.add(1)

    def _check_checkpoint_path(self, ckpt):
        """ Validate the checkpoint """
        return os.path.exists(ckpt)

    def restore(self, checkpoint_path=None):
        r"""
        Overview:
            restore learner from checkpoint_path
        Arguments:
            - checkpoint_path (:obj:`str`): the checkpoint path to load from, if None then set to cfg.common.load_path
        """
        checkpoint_path = checkpoint_path or self.cfg.common.load_path
        ckpt_ok = self._check_checkpoint_path(checkpoint_path)
        if ckpt_ok:
            self.checkpoint_manager.load(
                checkpoint_path,
                self.agent.get_model(),
                optimizer=self.optimizer,
                last_frame=self.last_frame,
                last_iter=self.last_iter,
                last_epoch=self.last_epoch,  # TODO last_epoch for lr_scheduler
                dataset=self.ckpt_dataset,
                logger_prefix='({})'.format(self._name)
            )
        self.last_frame.update(int(self.last_frame.val / self.world_size))  # adjust to different GPUs

    def save_checkpoint(self):
        r"""
        Overview:
            save checkpoint named by current iteration(only rank 0)
        """
        if self.rank == 0:
            self.checkpoint_manager.save_iterations(
                self.last_iter.val,
                self.agent.get_model(),
                optimizer=self.optimizer,
                # dataset=self.dataset,
                dataset=None,
                last_epoch=self.last_epoch.val,
                last_frame=self.last_frame.val * self.world_size  # total frames from all GPUs
            )

    @auto_checkpoint
    def run(self):
        r"""
        Overview:
            train / evaluate model with dataset in numbers of epoch, main loop of Learner,
            and will automatically save checkpoints periodically
            wrapped by auto_checkpoint in checkpoint_helper, you can reference checkpoint_helper.auto_checkpoint
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
        """
        Overview:
            the pipeline of one iteration(data prepare, forward, backward)
        """
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
            # periodically evaluate
            if self.last_iter.val % self.cfg.logger.eval_freq == 0:
                self.evaluate()

    def _manage_learning_information(self, var_items, time_items):
        """
        Overview:
            manage the information produced by learning iteration, such as loss/criterion, time and other viz info
        Arguments:
            - var_items (:obj:`dict`) var_items: other variable items(e.g.: loss, acc)
            - time_items (:obj:`dict`) time_items: time items
        """

        # if use multi-GPU training, you should first aggregate these items among all the ranks
        if self.use_distributed:
            var_items = aggregate(var_items)
            time_items = aggregate(time_items)

        if self.rank != 0:
            return

        keys = list(self.variable_record.get_var_names('scalar')) + list(self.variable_record.get_var_names('1darray'))
        self.variable_record.update_var(transform_dict(var_items, keys))
        self.variable_record.update_var(time_items)

        iterations = self.last_iter.val
        total_frames = self.last_frame.val * self.world_size
        total_frames -= total_frames % 100
        # periodically print training info
        if iterations % self.cfg.logger.print_freq == 0:
            self.logger.info("=== Training Iteration {} Result ===".format(self.last_iter.val))
            self.logger.info('iterations:{}\t{}'.format(iterations, self.variable_record.get_vars_text()))
            tb_keys = self.tb_logger.scalar_var_names
            self.tb_logger.add_val_list(
                self.variable_record.get_vars_tb_format(tb_keys, total_frames, var_type='scalar'), viz_type='scalar'
            )
            self._record_additional_info(iterations)

        # periodically save checkpoint
        if iterations % self.cfg.logger.save_freq == 0:
            self.save_checkpoint()

    def finalize(self):
        r"""
        Overview:
            finalize learner at the end of training, used to clean sources such as finalizing linklink if used
            distributed
        """
        if self.use_distributed:
            dist_finalize()


class SupervisedLearner(Learner):
    r"""
    Overview:
        An abstract supervised learning learner class
    """
    _name = "BaseSupervisedLearner"


def transform_dict(var_items, keys):
    r"""
    Overview:
        transform a dict's certain key's tensor value into item or list type, and return the transformed dict
    Arguments:
        - var_items (:obj:`dict`): dict of var_items, value of with might be tensors
        - keys (:obj:`str`): keys of new_dict to return
    Returns:
        - new_dict (:obj:`dict`): the transformed dict
    """
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
    r"""
    Overview:
        aggregate the information from all ranks(usually use sync allreduce)
    Arguments:
        - data (:obj:`dict`): data needs to be reduced. Could be dict, torch.Tensor, numbers.Integral or numbers.Real.
    Returns:
        - new_data (:obj:`dict`): data after reduce
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
