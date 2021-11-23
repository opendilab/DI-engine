from typing import Optional
import copy
from easydict import EasyDict
import numpy as np

from ding.worker.buffer import DequeBuffer
from ding.worker.buffer.middleware import use_time_check
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
    )

    def __init__(
            self,
            cfg: EasyDict,
            tb_logger: Optional[object] = None,
            exp_name: str = 'default_experiement',
            instance_name: str = 'buffer'
    ) -> None:
        self.cfg = cfg
        self.name = '{}_iter'.format(instance_name)
        self.tb_logger = tb_logger
        self.buffer = DequeBuffer(size=cfg.replay_buffer_size)
        self.last_log_train_iter = -1

        # use_count middleware
        if self.cfg.max_use != float("inf"):
            self.buffer.use(use_time_check(self.buffer, max_use=self.cfg.max_use))

    def sample(self, size: int, train_iter: int):
        output = self.buffer.sample(size=size, ignore_insufficient=True)
        if len(output) > 0:
            if self.last_log_train_iter == -1 or train_iter - self.last_log_train_iter >= self.cfg.train_iter_per_log:
                meta = [o.meta for o in output]
                if self.cfg.max_use != float("inf"):
                    use_count_avg = np.mean([m['use_count'] for m in meta])
                    self.tb_logger.add_scalar('{}/use_count_avg'.format(self.name), use_count_avg, train_iter)
                self.tb_logger.add_scalar('{}/buffer_data_count'.format(self.name), self.buffer.count(), train_iter)
            return [o.data for o in output]
        else:
            return None

    def push(self, data, cur_collector_envstep: int = -1) -> None:
        # meta = {'train_iter_data_collected': }
        for d in data:
            self.buffer.push(d)
