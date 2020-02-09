from threading import Thread
import zmq
import time
import numpy as np
import torch
from collections import deque
from sc2learner.dataset import OnlineDataset, OnlineDataLoader
from sc2learner.utils import build_logger, build_checkpoint_helper, build_time_helper, to_device, CountVar,\
    DistributionTimeImage
from sc2learner.nn_utils import build_grad_clip


def build_optimizer(model, cfg):
    optimizer = torch.optim.Adam(model.parameters(), float(cfg.train.learning_rate))
    return optimizer


def build_lr_scheduler(optimizer):
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000], gamma=1)
    return lr_scheduler


class HistoryActorInfo(object):
    def __init__(self, actor_monitor_arg):
        self._data = {}
        self.actor_monitor_arg = actor_monitor_arg
        self.copy_keys = ['update_model_time', 'data_rollout_time']
        self.dist_time_img = {k: DistributionTimeImage() for k in self.copy_keys}
        self.game_results = {k: deque(maxlen=actor_monitor_arg.difficulty_queue_len)
                             for k in actor_monitor_arg.difficulties}

    def update_actor_info(self, data):
        actor_id = data['actor_id']
        if actor_id in self._data.keys():
            self._data[actor_id]['count'] += 1
            self._data[actor_id]['update_time'] = time.time()
        else:
            self._data[actor_id] = {'count': 1, 'update_time': time.time()}
        for k in self.copy_keys:
            self._data[actor_id][k] = data[k]
        if len(data['episode_infos']) > 0:
            game_result, difficulty = data['episode_infos'][0]['game_result'], data['episode_infos'][0]['difficulty']
            self.game_results[difficulty].append(game_result)

    def __str__(self):
        cur_time = time.time()
        s = ""
        for k, v in self._data.items():
            s += "actor_id({})".format(k)
            for k1, v1 in v.items():
                if k1 == 'update_time':
                    s += '\t{}({:.3f})'.format(k1, cur_time - v1)
                else:
                    s += '\t{}({})'.format(k1, v1)
            s += "\n"
        return s

    def get_cls_by_time(self):
        def monotonic_check(item, judge_type='increase_strict'):
            judge_func = {
                'increase_strict': lambda x, y: x >= y,
            }
            if judge_type in judge_func.keys():
                judge = judge_func[judge_type]
            else:
                raise NotImplementedError("invalid judge type: {}".format(judge_type))

            for i in range(len(item) - 1):
                if judge(item[i], item[i+1]):
                    return False
            return True

        keys = list(self.actor_monitor_arg.speed.keys())
        values = list(self.actor_monitor_arg.speed.values())
        assert(monotonic_check(values))

        def look_up(t):
            for idx, (k, v) in enumerate(zip(keys, values)):
                if t <= v:
                    return k
            return 'dead'

        cur_time = time.time()
        keys.extend(['dead', 'total'])
        result = {k: 0 for k in keys}
        for k, v in self._data.items():
            for k1, v1 in v.items():
                if k1 == 'update_time':
                    last_update_time = cur_time - v1
                    cls = look_up(last_update_time)
                    result[cls] += 1
                    result['total'] += 1
        return result

    def get_distribution(self, key):
        data = []
        for k, v in self._data.items():
            data.append(v[key])
        if len(data) == 0:
            data = [0]  # default when self._data is empty
        data = np.array(data)
        data = np.sort(data)
        return data

    def get_distribution_img(self, key):
        self.dist_time_img[key].add_one_time_step(self.get_distribution(key))
        return self.dist_time_img[key].get_image()

    def get_win_rate(self):
        def win_rate(v):
            return (sum(v) / (len(v) + 1e-6) + 1) / 2

        return {k: win_rate(v) for k, v in self.game_results.items()}

    def state_dict(self):
        return {'actor_data': self._data,
                'game_results': self.game_results}

    def load_state_dict(self, state_dict):
        self._data = state_dict['actor_data']
        self.game_results = state_dict['game_results']


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
                                     transform=self._data_transform, block_data=cfg.train.block_data,
                                     min_update_count=cfg.train.min_update_count)
        self.dataloader = OnlineDataLoader(self.dataset, batch_size=cfg.train.batch_size)

        ip = cfg.communication.ip
        port = cfg.communication.port
        pull_port = port['learner']
        rep_port = port['learner_manager_model']
        self.HWM = cfg.communication.HWM['learner']
        self.pull_thread = Thread(target=self._pull_data,
                                  args=(self.zmq_context, pull_port))
        self.reply_model_thread = Thread(target=self._reply_model,
                                         args=(self.zmq_context, rep_port))

        self.optimizer = build_optimizer(model, cfg)
        self.lr_scheduler = build_lr_scheduler(self.optimizer)
        self.logger, self.tb_logger, self.scalar_record = build_logger(cfg)
        self.grad_clipper = build_grad_clip(cfg)
        self.time_helper = build_time_helper(cfg)
        self.checkpoint_helper = build_checkpoint_helper(cfg)
        self.last_iter = CountVar(init_val=0)
        self.history_actor_info = HistoryActorInfo(cfg.logger.actor_monitor)
        if cfg.common.load_path != '':
            self.checkpoint_helper.load(cfg.common.load_path, self.model,
                                        optimizer=self.optimizer,
                                        last_iter=self.last_iter,  # TODO last_iter for lr_scheduler
                                        dataset=self.dataset,
                                        actor_info=self.history_actor_info,
                                        logger_prefix='(learner)')
        self._init()
        self._optimize_step = self.time_helper.wrapper(self._optimize_step)

    def run(self):
        self.pull_thread.start()
        self.reply_model_thread.start()
        while not self.dataset.is_full():
            print('Waiting...' + self.dataset.format_len())
            time.sleep(10)

        while True:
            self.lr_scheduler.step()
            cur_lr = self.lr_scheduler.get_lr()[0]
            self.time_helper.start_time()
            self.dataloader.cur_model_index = self.last_iter.val
            batch_data, avg_usage, push_count, avg_model_index = next(self.dataloader)
            if self.use_cuda:
                batch_data = to_device(batch_data, 'cuda')
            data_time = self.time_helper.end_time()
            var_items, forward_time = self._get_loss(batch_data)
            _, backward_update_time = self._optimize_step(var_items['total_loss'])
            time_items = {'data_time': data_time, 'forward_time': forward_time,
                          'backward_update_time': backward_update_time,
                          'total_batch_time': data_time+forward_time+backward_update_time}
            var_items['cur_lr'] = cur_lr
            var_items['avg_usage'] = avg_usage
            var_items['push_count'] = push_count
            var_items['data_staleness'] = self.last_iter.val - avg_model_index
            var_items['push_rate'] = push_count / (data_time + forward_time + backward_update_time)

            self._update_monitor_var(var_items, time_items)
            self._record_info(self.last_iter.val)
            self.last_iter.add(1)

    def _record_info(self, iterations):
        if iterations % self.cfg.logger.print_freq == 0:
            self.logger.info('iterations:{}\t{}'.format(iterations, self.scalar_record.get_var_all()))
            tb_keys = self.tb_logger.scalar_var_names
            self.tb_logger.add_scalar_list(self.scalar_record.get_var_tb_format(tb_keys, iterations))
            self.tb_logger.add_text('history_actor_info', str(self.history_actor_info), iterations)
            self.tb_logger.add_scalars('actor_monitor', self.history_actor_info.get_cls_by_time(), iterations)
            self.tb_logger.add_scalars('win_rate', self.history_actor_info.get_win_rate(), iterations)
            self.tb_logger.add_histogram('data_rollout_time',
                                         self.history_actor_info.get_distribution('data_rollout_time'), iterations)
            self.tb_logger.add_histogram('update_model_time',
                                         self.history_actor_info.get_distribution('update_model_time'), iterations)
            self.tb_logger.add_histogram('dataset_staleness',
                                         np.array([iterations - d['model_index']
                                                   for d in self.dataset.data_queue]), iterations)
            rollout_img = self.history_actor_info.get_distribution_img('data_rollout_time')
            self.tb_logger.add_image('data_rollout_time_img', rollout_img, iterations)
            update_model_img = self.history_actor_info.get_distribution_img('update_model_time')
            self.tb_logger.add_image('update_model_time_img', update_model_img, iterations)
        if iterations % self.cfg.logger.save_freq == 0:
            self.checkpoint_helper.save_iterations(iterations, self.model, optimizer=self.optimizer,
                                                   dataset=self.dataset, actor_info=self.history_actor_info)

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
        receiver.setsockopt(zmq.RCVHWM, self.HWM)
        receiver.setsockopt(zmq.SNDHWM, self.HWM)
        receiver.bind("tcp://*:%s" % (port))
        while True:
            data = receiver.recv_pyobj()
            if isinstance(data, dict):
                self.history_actor_info.update_actor_info(data)
                self._parse_pull_data(data)
            elif isinstance(data, list):
                for d in data:
                    self.history_actor_info.update_actor_info(d)
                    self._parse_pull_data(d)
            else:
                raise TypeError(type(data))

    def _parse_pull_data(self):
        raise NotImplementedError

    def _data_transform(self, data):
        raise NotImplementedError

    def _reply_model(self, zmq_context, port):
        receiver = zmq_context.socket(zmq.DEALER)
        receiver.bind("tcp://*:%s" % (port))
        while True:
            msg = receiver.recv_string()
            assert(msg == 'request model')
            state_dict = {k: v.to('cpu') for k, v in self.model.state_dict().items()}
            state_dict = {'state_dict': state_dict, 'model_index': self.last_iter.val}
            receiver.send_pyobj(state_dict)

    def _init(self):
        self.scalar_record.register_var('cur_lr')
        self.scalar_record.register_var('push_rate')
        self.scalar_record.register_var('avg_usage')
        self.scalar_record.register_var('push_count')
        self.scalar_record.register_var('data_staleness')
        self.scalar_record.register_var('total_batch_time')
        self.scalar_record.register_var('data_time')
        self.scalar_record.register_var('forward_time')
        self.scalar_record.register_var('backward_update_time')
        self.scalar_record.register_var('backward_time')
        self.scalar_record.register_var('grad_clipper_time')
        self.scalar_record.register_var('update_step_time')

        self.tb_logger.register_var('cur_lr')
        self.tb_logger.register_var('push_rate')
        self.tb_logger.register_var('avg_usage')
        self.tb_logger.register_var('push_count')
        self.tb_logger.register_var('data_staleness')
        self.tb_logger.register_var('total_batch_time')
        self.tb_logger.register_var('history_actor_info', var_type='text')
        self.tb_logger.register_var('actor_monitor', var_type='scalars')
        self.tb_logger.register_var('data_rollout_time', var_type='histogram')
        self.tb_logger.register_var('update_model_time', var_type='histogram')
        self.tb_logger.register_var('dataset_staleness', var_type='histogram')
        self.tb_logger.register_var('data_rollout_time_img', var_type='image')
        self.tb_logger.register_var('update_model_time_img', var_type='image')
        self.tb_logger.register_var('win_rate', var_type='scalars')
