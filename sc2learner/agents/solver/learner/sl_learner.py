'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. base class for supervised learning on linklink, including basic processes.
'''
import os
import numbers
import torch
from sc2learner.dataset import ReplayDataset, ReplayEvalDataset, build_dataloader
from sc2learner.utils import build_logger, build_checkpoint_helper, build_time_helper, to_device, CountVar,\
    DistModule, dist_init, dist_finalize, allreduce, auto_checkpoint
from sc2learner.agents.model import build_model


def build_optimizer(model, cfg):
    '''
        Overview: use config to initialize optimizer. Use Adam by default.
        Arguments:
            - model (:obj:`torch.nn.Module`): model with param
            - cfg (:obj:`dict`): optimizer config
        Returns:
            - (:obj`Optimizer`): optimizer created by this function
    '''
    optimizer = torch.optim.Adam(model.parameters(), float(cfg.train.learning_rate),
                                 weight_decay=float(cfg.train.weight_decay))
    return optimizer


def build_lr_scheduler(optimizer):
    '''
        Overview: use optimizer to build lr scheduler. Use MultiStepLR by default.
        Arguments:
            - optimizer (:obj:`Optimizer`): Optimizer need by scheduler
        Returns:
            - (:obj`lr_scheduler`): lr_scheduler created by this function
    '''
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000], gamma=1)
    return lr_scheduler


class SLLearner(object):
    '''
        Overview: base class for supervised learning on linklink, including basic processes.
        Interface: __init__, run, finalize, save_checkpoint, eval
    '''

    def __init__(self, cfg=None):
        '''
            Overview: initialization method, using setting to build model, dataset, optimizer, lr_scheduler
                      and other helper. It can alse load checkpoint.
            Arguments:
                - cfg (:obj:`dict`): learner config
        '''
        assert(cfg is not None)
        # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # for debug async CUDA
        self.cfg = cfg
        self.use_distributed = cfg.train.use_distributed
        if self.use_distributed:
            self.rank, self.world_size = dist_init()  # initialize rank and world size for linklink
        else:
            self.rank, self.world_size = 0, 1
        self.model = build_model(cfg)  # build model by policy from alphaStar
        self.model.train()  # set model to train
        self.use_cuda = cfg.train.use_cuda
        if self.use_cuda:
            self.model = to_device(self.model, 'cuda')
        if self.use_distributed:
            self.model = DistModule(self.model)  # distributed training

        self.dataset = ReplayDataset(cfg.data.train)  # use replay as dataset
        self.eval_dataset = ReplayEvalDataset(cfg.data.eval)
        self.dataloader = build_dataloader(cfg.data.train, self.dataset)
        self.eval_dataloader = build_dataloader(cfg.data.eval, self.eval_dataset)
        self.train_dataloader_type = cfg.data.train.dataloader_type
        assert(self.train_dataloader_type in ['epoch', 'iter'])

        self.optimizer = build_optimizer(self.model, cfg)  # build optimizer using cfg
        self.lr_scheduler = build_lr_scheduler(self.optimizer)  # build lr_scheduler
        if self.rank == 0:  # only one thread need to build logger
            self.logger, self.tb_logger, self.variable_record = build_logger(cfg, rank=self.rank)
            self.logger.info('cfg:\n{}'.format(self.cfg))
            self.logger.info('model:\n{}'.format(self.model))
            self._init()
        else:
            self.logger, _, _ = build_logger(cfg, rank=self.rank)

        self.time_helper = build_time_helper(cfg)  # build time_helper for timing
        self.checkpoint_helper = build_checkpoint_helper(cfg, self.rank)  # build checkpoint_helper to load or save
        self.last_iter = CountVar(init_val=0)  # count for iterations
        self.last_epoch = CountVar(init_val=0)  # count for epochs

        if self.train_dataloader_type == 'epoch':
            ckpt_dataset = self.dataset
        elif self.train_dataloader_type == 'iter':
            # iter type doesn't save some context
            ckpt_dataset = None
        if cfg.common.load_path != '':
            self.checkpoint_helper.load(cfg.common.load_path, self.model,
                                        optimizer=self.optimizer,
                                        last_iter=self.last_iter,
                                        last_epoch=self.last_epoch,  # TODO last_epoch for lr_scheduler
                                        dataset=ckpt_dataset,
                                        logger_prefix='(sl_learner)')
            self.last_epoch.add(1)  # skip interrupted epoch
            self.last_iter.add(1)  # skip interrupted iter
        self._optimize_step = self.time_helper.wrapper(self._optimize_step)
        self.max_epochs = cfg.train.max_epochs

    @auto_checkpoint
    def run(self):
        '''
            Overview: train/evaluate model with dataset in numbers of epoch
        '''
        if self.cfg.common.only_evaluate:
            self.eval()
            return

        def train_epoch():
            for idx, data in enumerate(self.dataloader):  # one epoch
                self.time_helper.start_time()
                data_stat = self._get_data_stat(data)
                if self.use_cuda:
                    batch_data = to_device(data, 'cuda')
                data_time = self.time_helper.end_time()  # cal data load time
                var_items, forward_time = self._get_loss(batch_data)  # train process
                _, backward_update_time = self._optimize_step(var_items['total_loss'])  # process loss
                time_items = {'data_time': data_time, 'forward_time': forward_time,
                              'backward_update_time': backward_update_time,
                              'total_batch_time': data_time+forward_time+backward_update_time}
                var_items['cur_lr'] = self.lr_scheduler.get_lr()[0]
                var_items['epoch'] = self.last_epoch.val
                var_items.update(data_stat)

                if self.use_distributed:
                    var_items, time_items = [self._reduce_info(x) for x in [var_items, time_items]]
                if self.rank == 0:
                    self._update_monitor_var(var_items, time_items)  # update monitor variables
                    self._record_info(self.last_iter.val)  # save logger info
                self.last_iter.add(1)
                if self.last_iter.val % self.cfg.logger.eval_freq == 0:
                    self.eval()

        if self.train_dataloader_type == 'epoch':
            while self.last_epoch.val < self.max_epochs:
                if hasattr(self.dataloader.dataset, 'step'):  # call dataset.step()
                    self.dataloader.dataset.step()
                self.lr_scheduler.step()  # update lr
                train_epoch()
                self.last_epoch.add(1)
        elif self.train_dataloader_type == 'iter':
            train_epoch()
        # save the final checkpoint
        self.save_checkpoint()

    def eval(self):
        raise NotImplementedError

    def finalize(self):
        '''
            Overview: finalize, called after training
        '''
        if self.use_distributed:
            dist_finalize()

    def save_checkpoint(self):
        '''
            Overview: save checkpoint named by current iteration(only rank 0)
        '''
        if self.rank == 0:
            self.checkpoint_helper.save_iterations(self.last_iter.val, self.model, optimizer=self.optimizer,
                                                   dataset=self.dataset, last_epoch=self.last_epoch.val)

    def _get_data_stat(self, data):
        '''
            Overview: empty interface for data statistics
            Arguments:
                - data (:obj:`dict`): data dict for one step iteration
            Returns:
                - (:obj`dict`): data statistics(default empty dict)
        '''
        return {}

    def _reduce_info(self, data):
        '''
            Overview: merge all info from other rank
            Arguments:
                - data (:obj:`dict`): data needs to be reduced. Could be dict, torch.Tensor,
                                      numbers.Integral or numbers.Real
            Returns:
                - (:obj`dict`): data after reduce
        '''
        if isinstance(data, dict):
            new_data = {}
            for k, v in data.items():
                new_data[k] = self._reduce_info(v)
        elif isinstance(data, list):
            new_data = []
            for t in data:
                new_data.append(self._reduce_info(t))
        elif isinstance(data, torch.Tensor):
            new_data = data.clone()
            allreduce(new_data)  # get data from other processes
            new_data.div_(self.world_size)  # get average on all ranks
        elif isinstance(data, numbers.Integral) or isinstance(data, numbers.Real):
            new_data = torch.Tensor([data])
            allreduce(new_data)
            new_data = new_data.item() / self.world_size
        else:
            raise TypeError("invalid info type: {}".format(type(data)))
        return new_data

    def _record_info(self, iterations):
        '''
            Overview: record on logger or save checkpoint
            Arguments:
                - iterations (:obj:`int`): iteration number
        '''
        if iterations % self.cfg.logger.print_freq == 0:
            self.logger.info('iterations:{}\t{}'.format(iterations, self.variable_record.get_vars_text()))
            tb_keys = self.tb_logger.scalar_var_names
            self.tb_logger.add_val_list(self.variable_record.get_vars_tb_format(
                tb_keys, iterations, var_type='scalar'), viz_type='scalar')
            self._record_additional_info(iterations)
        if iterations % self.cfg.logger.save_freq == 0:
            self.checkpoint_helper.save_iterations(iterations, self.model, optimizer=self.optimizer,
                                                   dataset=self.dataset, last_epoch=self.last_epoch.val)

    def _get_loss(self, data):
        '''
            Overview: main process of training
            Arguments:
                - data (:obj:`batch_data`): batch_data created by dataloader
        '''
        raise NotImplementedError

    def _record_additional_info(self, iterations):
        '''
            Overview: empty interface to record additional info on logger
            Arguments:
                - iterations (:obj:`int`): iteration number
        '''
        pass

    def _update_monitor_var(self, var_items, time_items):
        '''
            Overview: update monitor variables by given keys
            Arguments:
                - var_items (:obj:`dict`): use loss keys to update certain variables
                - time_items (:obj:`dict`): time items need to be updated
        '''
        keys = list(self.variable_record.get_var_names('scalar')) + list(self.variable_record.get_var_names('1darray'))
        new_dict = {}
        for k in keys:
            if k in var_items.keys():
                v = var_items[k]
                if isinstance(v, torch.Tensor):
                    if v.shape == (1,):
                        v = v.item()  # get item
                    else:
                        v = v.tolist()
                else:
                    v = v
                new_dict[k] = v
        self.variable_record.update_var(new_dict)
        self.variable_record.update_var(time_items)

    def _optimize_step(self, loss):
        '''
            Overview: update by optimizer
            Arguments:
                - loss (:obj:`tensor`): loss to be backward.
        '''
        self.optimizer.zero_grad()
        avg_loss = loss / self.world_size
        avg_loss.backward()
        if self.use_distributed:
            self.model.sync_gradients()
        self.optimizer.step()

    def _init(self):
        '''
            Overview: initialize logger
        '''
        self.variable_record.register_var('cur_lr')
        self.variable_record.register_var('epoch')
        self.variable_record.register_var('data_time')
        self.variable_record.register_var('forward_time')
        self.variable_record.register_var('backward_update_time')
        self.variable_record.register_var('total_batch_time')
        self.tb_logger.register_var('cur_lr')
        self.tb_logger.register_var('epoch')
        self.tb_logger.register_var('total_batch_time')
