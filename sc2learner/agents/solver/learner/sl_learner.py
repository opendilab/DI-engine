import torch
from torch.utils.data import DataLoader
from sc2learner.dataset import ReplayDataset
from sc2learner.utils import build_logger, build_checkpoint_helper, build_time_helper, to_device, CountVar
from sc2learner.agents.model import build_model


def build_optimizer(model, cfg):
    optimizer = torch.optim.Adam(model.parameters(), float(cfg.train.learning_rate),
                                 weight_decay=float(cfg.train.weight_decay))
    return optimizer


def build_lr_scheduler(optimizer):
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000], gamma=1)
    return lr_scheduler


class SLLearner(object):

    def __init__(self, cfg=None):
        assert(cfg is not None)
        self.cfg = cfg
        self.model = build_model(cfg)
        print(self.model)
        self.model.train()
        self.use_cuda = cfg.train.use_cuda
        if self.use_cuda:
            self.model = to_device(self.model, 'cuda')
        self.dataset = ReplayDataset(cfg.data.replay_list, cfg.data.trajectory_len, cfg.data.trajectory_type)
        self.dataloader = DataLoader(self.dataset, batch_size=cfg.train.batch_size, pin_memory=False, num_workers=3,
                                     shuffle=True, drop_last=True)

        self.optimizer = build_optimizer(self.model, cfg)
        self.lr_scheduler = build_lr_scheduler(self.optimizer)
        self.logger, self.tb_logger, self.scalar_record = build_logger(cfg)
        self.time_helper = build_time_helper(cfg)
        self.checkpoint_helper = build_checkpoint_helper(cfg)
        self.last_iter = CountVar(init_val=0)
        if cfg.common.load_path != '':
            self.checkpoint_helper.load(cfg.common.load_path, self.model,
                                        optimizer=self.optimizer,
                                        last_iter=self.last_iter,  # TODO last_iter for lr_scheduler
                                        dataset=self.dataset,
                                        logger_prefix='(sl_learner)')
        self._init()
        self._optimize_step = self.time_helper.wrapper(self._optimize_step)
        self.max_epochs = cfg.train.max_epochs

    def run(self):

        for epoch in range(self.max_epochs):
            if hasattr(self.dataloader.dataset, 'step'):
                self.dataloader.dataset.step()
            self.lr_scheduler.step()
            cur_lr = self.lr_scheduler.get_lr()[0]
            for idx, data in enumerate(self.dataloader):
                self.time_helper.start_time()
                if self.use_cuda:
                    batch_data = to_device(data, 'cuda')
                data_time = self.time_helper.end_time()
                var_items, forward_time = self._get_loss(batch_data)
                _, backward_update_time = self._optimize_step(var_items['total_loss'])
                time_items = {'data_time': data_time, 'forward_time': forward_time,
                              'backward_update_time': backward_update_time}
                print(time_items)
                var_items['cur_lr'] = cur_lr
                var_items['epoch'] = epoch

                self._update_monitor_var(var_items, time_items)
                self._record_info(self.last_iter.val)
                self.last_iter.add(1)

    def _record_info(self, iterations):
        if iterations % self.cfg.logger.print_freq == 0:
            self.logger.info('iterations:{}\t{}'.format(iterations, self.scalar_record.get_var_all()))
            tb_keys = self.tb_logger.scalar_var_names
            self.tb_logger.add_scalar_list(self.scalar_record.get_var_tb_format(tb_keys, iterations))
        if iterations % self.cfg.logger.save_freq == 0:
            self.checkpoint_helper.save_iterations(iterations, self.model, optimizer=self.optimizer,
                                                   dataset=self.dataset)

    def _get_loss(self, data):
        raise NotImplementedError

    def _update_monitor_var(self, loss_items, time_items):
        keys = self.scalar_record.get_var_names()
        new_dict = {}
        for k in keys:
            if k in loss_items.keys():
                v = loss_items[k]
                if isinstance(v, torch.Tensor):
                    v = v.item()
                else:
                    v = v
                new_dict[k] = v
        self.scalar_record.update_var(new_dict)
        self.scalar_record.update_var(time_items)

    def _optimize_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        # TODO support reduce gradient
        self.optimizer.step()

    def _init(self):
        self.scalar_record.register_var('cur_lr')
        self.scalar_record.register_var('epoch')
        self.scalar_record.register_var('data_time')
        self.scalar_record.register_var('forward_time')
        self.scalar_record.register_var('backward_update_time')
        self.scalar_record.register_var('total_batch_time')
        self.tb_logger.register_var('cur_lr')
        self.tb_logger.register_var('epoch')
        self.tb_logger.register_var('total_batch_time')
