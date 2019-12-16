from threading import Thread
import zmq
import time
import torch
from sc2learner.dataset import OnlineDataset, OnlineDataLoader
from sc2learner.utils import build_logger, build_checkpoint_helper, build_time_helper, to_device, CountVar
from sc2learner.nn_utils import build_grad_clip


def build_optimizer(model, cfg):
    optimizer = torch.optim.Adam(model.parameters(), float(cfg.train.learning_rate))
    return optimizer


def build_lr_scheduler(optimizer):
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000], gamma=1)
    return lr_scheduler


class BaseLearner(object):

    def __init__(self, env, model, cfg=None):
        assert(cfg is not None)
        self.cfg = cfg
        self.env = env
        self.model = model
        self.model.train()
        self.use_cuda = cfg.train.learner_use_cuda
        if self.use_cuda:
            self.model = to_device(self.model, 'cuda')
        self.unroll_length = cfg.train.unroll_length

        self.zmq_context = zmq.Context()
        self.dataset = OnlineDataset(data_maxlen=cfg.train.learner_data_queue_size,
                                     episode_maxlen=cfg.train.learner_episode_queue_size,
                                     transform=self._data_transform)
        self.dataloader = OnlineDataLoader(self.dataset, batch_size=cfg.train.batch_size)
        port = cfg.communication.port
        self.pull_thread = Thread(target=self._pull_data,
                                  args=(self.zmq_context, port['actor']))
        self.reply_model_thread = Thread(target=self._reply_model,
                                         args=(self.zmq_context, port['learner']))

        self.optimizer = build_optimizer(model, cfg)
        self.lr_scheduler = build_lr_scheduler(self.optimizer)
        self.logger, self.tb_logger, self.scalar_record = build_logger(cfg)
        self.grad_clipper = build_grad_clip(cfg)
        self.time_helper = build_time_helper(cfg)
        self.checkpoint_helper = build_checkpoint_helper(cfg)
        self.last_iter = CountVar(init_val=0)
        if cfg.common.load_path != '':
            self.checkpoint_helper.load(cfg.common.load_path, self.model,
                                        optimizer=self.optimizer,
                                        last_iter=self.last_iter,  # TODO last_iter for lr_scheduler
                                        logger_prefix='(learner)')
        self._init()
        self._optimize_step = self.time_helper.wrapper(self._optimize_step)

    def run(self):
        self.pull_thread.start()
        self.reply_model_thread.start()
        while not self.dataset.is_episode_full():
            print('Waiting...' + self.dataset.episode_len())
            time.sleep(10)

        while True:
            self.lr_scheduler.step()
            cur_lr = self.lr_scheduler.get_lr()[0]
            self.time_helper.start_time()
            batch_data = next(self.dataloader)
            if self.use_cuda:
                batch_data = to_device(batch_data, 'cuda')
            data_time = self.time_helper.end_time()
            var_items, forward_time = self._get_loss(batch_data)
            _, backward_update_time = self._optimize_step(var_items['total_loss'])
            time_items = {'data_time': data_time, 'forward_time': forward_time,
                          'backward_update_time': backward_update_time}
            var_items['cur_lr'] = cur_lr

            self._update_monitor_var(var_items, time_items)
            self._record_info(self.last_iter.val)
            self.last_iter.add(1)

    def _record_info(self, iterations):
        if iterations % self.cfg.logger.print_freq == 0:
            self.logger.info('iterations:{}\t{}'.format(iterations, self.scalar_record.get_var_all()))
            tb_keys = self.tb_logger.scalar_var_names
            self.tb_logger.add_scalar_list(self.scalar_record.get_var_tb_format(tb_keys, iterations))
        if iterations % self.cfg.logger.save_freq == 0:
            self.checkpoint_helper.save_iterations(iterations, self.model, optimizer=self.optimizer)

    def _get_loss(self, data):
        raise NotImplementedError

    def _update_monitor_var(self, loss_items, time_items):
        raise NotImplementedError

    def _optimize_step(self, loss):
        self.optimizer.zero_grad()
        self.time_helper.start_time()
        loss.backward()
        backward_time = self.time_helper.end_time()
        self.time_helper.start_time()
        self.grad_clipper.apply(self.model.parameters())
        grad_clipper_time = self.time_helper.end_time()
        self.time_helper.start_time()
        # TODO support reduce gradient
        self.optimizer.step()
        update_step_time = self.time_helper.end_time()
        self.scalar_record.update_var({'backward_time': backward_time,
                                       'grad_clipper_time': grad_clipper_time,
                                       'update_step_time': update_step_time})

    def _pull_data(self, zmq_context, port):
        receiver = zmq_context.socket(zmq.PULL)
        receiver.setsockopt(zmq.RCVHWM, 1)
        receiver.setsockopt(zmq.SNDHWM, 1)
        receiver.bind("tcp://*:%s" % (port))
        while True:
            data = receiver.recv_pyobj()
            self._parse_pull_data(data)

    def _parse_pull_data(self):
        raise NotImplementedError

    def _data_transform(self, data):
        raise NotImplementedError

    def _reply_model(self, zmq_context, port):
        receiver = zmq_context.socket(zmq.REP)
        receiver.bind("tcp://*:%s" % (port))
        while True:
            msg = receiver.recv_string()
            assert(msg == 'request model')
            state_dict = {k: v.to('cpu') for k, v in self.model.state_dict().items()}
            receiver.send_pyobj(state_dict)

    def _init(self):
        self.scalar_record.register_var('cur_lr')
        self.scalar_record.register_var('data_time')
        self.scalar_record.register_var('forward_time')
        self.scalar_record.register_var('backward_update_time')
        self.scalar_record.register_var('backward_time')
        self.scalar_record.register_var('grad_clipper_time')
        self.scalar_record.register_var('update_step_time')
