from typing import Optional
import copy
from easydict import EasyDict
import numpy as np

from ding.data.buffer import DequeBuffer
from ding.data.buffer.middleware import use_time_check, PriorityExperienceReplay
from ding.utils import BUFFER_REGISTRY


@BUFFER_REGISTRY.register('deque')
class DequeBufferWrapper(object):

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        replay_buffer_size=10000,
        max_use=float("inf"),
        train_iter_per_log=100,
        priority=False,
        priority_IS_weight=False,
        priority_power_factor=0.6,
        IS_weight_power_factor=0.4,
        IS_weight_anneal_train_iter=int(1e5),
        priority_max_limit=1000,
    )

    def __init__(
            self,
            cfg: EasyDict,
            tb_logger: Optional[object] = None,
            exp_name: str = 'default_experiement',
            instance_name: str = 'buffer'
    ) -> None:
        self.cfg = cfg
        self.priority_max_limit = cfg.priority_max_limit
        self.name = '{}_iter'.format(instance_name)
        self.tb_logger = tb_logger
        self.buffer = DequeBuffer(size=cfg.replay_buffer_size)
        self.last_log_train_iter = -1

        # use_count middleware
        if self.cfg.max_use != float("inf"):
            self.buffer.use(use_time_check(self.buffer, max_use=self.cfg.max_use))
        # priority middleware
        if self.cfg.priority:
            self.buffer.use(
                PriorityExperienceReplay(
                    self.buffer,
                    self.cfg.replay_buffer_size,
                    IS_weight=self.cfg.priority_IS_weight,
                    priority_power_factor=self.cfg.priority_power_factor,
                    IS_weight_power_factor=self.cfg.IS_weight_power_factor,
                    IS_weight_anneal_train_iter=self.cfg.IS_weight_anneal_train_iter
                )
            )
            self.last_sample_index = None
            self.last_sample_meta = None

    def sample(self, size: int, train_iter: int = 0):
        output = self.buffer.sample(size=size, ignore_insufficient=True)
        if len(output) > 0:
            if self.last_log_train_iter == -1 or train_iter - self.last_log_train_iter >= self.cfg.train_iter_per_log:
                meta = [o.meta for o in output]
                if self.cfg.max_use != float("inf"):
                    use_count_avg = np.mean([m['use_count'] for m in meta])
                    self.tb_logger.add_scalar('{}/use_count_avg'.format(self.name), use_count_avg, train_iter)
                if self.cfg.priority:
                    self.last_sample_index = [o.index for o in output]
                    self.last_sample_meta = meta
                    priority_list = [m['priority'] for m in meta]
                    priority_avg = np.mean(priority_list)
                    priority_max = np.max(priority_list)
                    self.tb_logger.add_scalar('{}/priority_avg'.format(self.name), priority_avg, train_iter)
                    self.tb_logger.add_scalar('{}/priority_max'.format(self.name), priority_max, train_iter)
                self.tb_logger.add_scalar('{}/buffer_data_count'.format(self.name), self.buffer.count(), train_iter)
                self.last_log_train_iter = train_iter

            data = [o.data for o in output]
            if self.cfg.priority_IS_weight:
                IS = [o.meta['priority_IS'] for o in output]
                for i in range(len(data)):
                    data[i]['IS'] = IS[i]
            return data
        else:
            return None

    def push(self, data, cur_collector_envstep: int = -1) -> None:
        for d in data:
            meta = {}
            if self.cfg.priority and 'priority' in d:
                init_priority = d.pop('priority')
                meta['priority'] = init_priority
            self.buffer.push(d, meta=meta)

    def update(self, meta: dict) -> None:
        if not self.cfg.priority:
            return
        if self.last_sample_index is None:
            return
        new_meta = self.last_sample_meta
        for m, p in zip(new_meta, meta['priority']):
            m['priority'] = min(self.priority_max_limit, p)
        for idx, m in zip(self.last_sample_index, new_meta):
            self.buffer.update(idx, data=None, meta=m)
        self.last_sample_index = None
        self.last_sample_meta = None

    def count(self) -> int:
        return self.buffer.count()
