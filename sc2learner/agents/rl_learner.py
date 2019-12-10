from threading import Thread
from collections import deque
import zmq
import time
import torch
from sc2learner.agents.rl_dataloader import RLBaseDataset, RLBaseDataLoader, unroll_split_collate_fn
from sc2learner.utils import build_logger, build_checkpoint_helper, build_time_helper
from sc2learner.nn_utils import build_grad_clip


def build_optimizer(model, cfg):
    optimizer = torch.optim.Adam(model.parameters(), float(cfg.train.learning_rate))
    return optimizer


def build_lr_scheduler(optimizer):
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000], gamma=0.1)
    return lr_scheduler


def build_clip_range_scheduler(cfg):
    class NaiveClip(object):
        def __init__(self, init_val=0.1):
            self.init_val = init_val

        def step(self):
            return self.init_val

    return NaiveClip(init_val=cfg.train.ppo_clip_range)


class BaseLearner(object):

    def __init__(self, env, model, cfg=None):
        assert(cfg is not None)
        self.cfg = cfg
        self.env = env
        self.model = model
        self.unroll_length = cfg.train.unroll_length
        self.episode_infos = deque(maxlen=cfg.train.learner_episode_queue_size)

        self.zmq_context = zmq.Context()
        self.dataset = RLBaseDataset(maxlen=cfg.train.learner_data_queue_size, transform=self._data_transform)
        self.dataloader = RLBaseDataLoader(self.dataset, batch_size=cfg.train.batch_size)
        port = cfg.communication.port
        self.pull_thread = Thread(target=self._pull_data,
                                  args=(self.zmq_context, port['actor']))
        self.reply_model_thread = Thread(target=self._reply_model,
                                         args=(self.zmq_context, port['learner']))

        self.optimizer = build_optimizer(model, cfg)
        self.lr_scheduler = build_lr_scheduler(self.optimizer)
        self.logger, self.tb_logger, self.scalar_record = build_logger(cfg)
        self.grad_clipper = build_grad_clip(cfg)
        self.checkpoint_helper = build_checkpoint_helper(cfg)
        if cfg.common.load_path != '':
            self.checkpoint_helper.load(cfg.common.load_path, self.model,
                                        optimizer=self.optimizer,
                                        logger_prefix='(learner)')
        self._init()
        self.time_helper = build_time_helper(cfg)
        self._optimize_step = self.time_helper.wrapper(self._optimize_step)

    def run(self):
        self.pull_thread.start()
        self.reply_model_thread.start()
        while len(self.episode_infos) < self.episode_infos.maxlen // 2:
            print('current episode_infos len:{}'.format(len(self.episode_infos)))
            time.sleep(10)

        iterations = 0
        while True:
            self.lr_scheduler.step()
            self.time_helper.start_time()
            batch_data = next(self.dataloader)
            data_time = self.time_helper.end_time()
            loss_items, model_time = self._get_loss(batch_data)
            _, update_time = self._optimize_step(loss_items['total_loss'])
            time_items = {'data_time': data_time, 'model_time': model_time, 'update_time': update_time}
            self._update_monitor_var(loss_items, time_items)
            self._record_info(iterations)
            iterations += 1

    def _record_info(self, iterations):
        if iterations % self.cfg.logger.print_freq == 0:
            self.logger.info('iterations:{}\t{}'.format(iterations, self.scalar_record.get_var_all()))
        if iterations % self.cfg.logger.save_freq == 0:
            self.checkpoint_helper.save_iterations(iterations, self.model, optimizer=self.optimizer)

    #@time_helper.wrapper
    def _get_loss(self, data):
        raise NotImplementedError

    def _update_monitor_var(self, loss_items, time_items):
        raise NotImplementedError

    #@time_helper.wrapper
    def _optimize_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.grad_clipper.apply(self.model.parameters())
        # TODO support reduce gradient
        self.optimizer.step()

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
            receiver.send_pyobj(self.model.state_dict())

    def _save_checkpoint(self, path, model, optimizer=None, last_iter=None):
        checkpoint = {}
        checkpoint['state_dict'] = model.state_dict()
        if optimizer is not None:
            assert(last_iter is not None)
            checkpoint['last_iter'] = last_iter
            checkpoint['optimizer'] = optimizer.state_dict()
        torch.save(checkpoint, path)

    def _get_output_msg(self):
        raise NotImplementedError

    def _print2logger(self, strings):
        raise NotImplementedError

    def _init(self):
        raise NotImplementedError


class PpoLearner(BaseLearner):
    def __init__(self, *args, **kwargs):
        super(PpoLearner, self).__init__(*args, **kwargs)
        self.use_value_clip = self.cfg.train.use_value_clip
        self.unroll_split = self.cfg.train.unroll_split if self.model.initial_state is None else 1
        self.entropy_coeff = self.cfg.train.entropy_coeff
        self.value_coeff = self.cfg.train.value_coeff
        self.clip_range_scheduler = build_clip_range_scheduler(self.cfg)
        self.dataloader = RLBaseDataLoader(self.dataset, self.cfg.train.batch_size * self.unroll_split,
                                           collate_fn=unroll_split_collate_fn)
        self._get_loss = self.time_helper.wrapper(self._get_loss)

    # overwrite
    def _init(self):
        self.scalar_record.register_var('total_loss')
        self.scalar_record.register_var('pg_loss')
        self.scalar_record.register_var('value_loss')
        self.scalar_record.register_var('entropy_reg')
        self.scalar_record.register_var('approximate_kl')
        self.scalar_record.register_var('clipfrac')
        self.scalar_record.register_var('data_time')
        self.scalar_record.register_var('model_time')
        self.scalar_record.register_var('update_time')

    # overwrite
    def _parse_pull_data(self, data):
        if self.unroll_split > 1:  # TODO (speed and memory copy optimization)
            keys = ['obs', 'return', 'done', 'action', 'value', 'neglogp', 'state']
            if self.model.use_mask:
                keys.append('mask')
            temp_dict = {}
            for k, v in data.items():
                if k in keys:
                    if k == 'state':
                        if v is None:
                            temp_dict[k] = [None for _ in range(self.unroll_split)]
                        else:
                            raise NotImplementedError
                    else:
                        stack_item = torch.stack(v, dim=0)
                        split_item = torch.chunk(stack_item, self.unroll_split)
                        temp_dict[k] = split_item
            for i in range(self.unroll_split):
                item = {k: v[i] for k, v in temp_dict.items()}
                self.dataset.push(item)
        else:
            self.dataset.push(data)
        self.episode_infos.extend(data['episode_infos'])

    # overwrite
    def _get_loss(self, data):
        clip_range = self.clip_range_scheduler.step()
        obs, actions, returns, neglogps, values, dones, states = (
                data['obs'], data['action'], data['return'], data['neglogp'],
                data['value'], data['done'], data['state']
            )

        inputs = {}
        inputs['obs'] = obs
        inputs['done'] = dones
        inputs['state'] = states
        if self.model.use_mask:
            inputs['mask'] = data['mask']
        new_values = self.model(inputs, mode='step')[1]
        new_neglogp = self.model.pd.neglogp(actions, reduction='none')
        entropy = self.model.pd.entropy().mean()

        new_values_clipped = values + torch.clamp(new_values - values, -clip_range, clip_range)
        value_loss1 = (new_values - returns) ** 2
        if self.use_value_clip:
            value_loss2 = torch.pow(new_values_clipped - returns, 2)
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        else:
            value_loss = 0.5 * value_loss1.mean()

        adv = returns - values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ratio = torch.exp(neglogps - new_neglogp)
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        approximate_kl = 0.5 * torch.pow(new_neglogp - neglogps, 2).mean()
        clipfrac = torch.abs(ratio - 1.0).gt(clip_range).float().mean()

        loss = pg_loss - entropy * self.entropy_coeff + value_loss * self.value_coeff
        loss_items = {}
        loss_items['total_loss'] = loss
        loss_items['approximate_kl'] = approximate_kl
        loss_items['clipfrac'] = clipfrac
        loss_items['pg_loss'] = pg_loss
        loss_items['entropy_reg'] = entropy * self.entropy_coeff
        loss_items['value_loss'] = value_loss * self.value_coeff
        return loss_items

    # overwrite
    def _update_monitor_var(self, items, time_items):
        keys = self.scalar_record.get_var_names()
        new_dict = {}
        for k in keys:
            if k in items.keys():
                v = items[k]
                if isinstance(v, torch.Tensor):
                    v = v.item()
                else:
                    v = v
                new_dict[k] = v
        self.scalar_record.update_var(new_dict)
        self.scalar_record.update_var(time_items)

    # overwrite
    def _data_transform(self, data):
        keys = ['obs', 'return', 'done', 'action', 'value', 'neglogp', 'state']
        if self.model.use_mask:
            keys.append('mask')
        transformed_data = {}
        for k, v in data.items():
            if k in keys:
                if k == 'state' and v is None:
                    transformed_data[k] = 'none'
                else:
                    transformed_data[k] = v
        return transformed_data
