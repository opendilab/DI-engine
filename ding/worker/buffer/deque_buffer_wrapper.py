from typing import Optional
import copy
from easydict import EasyDict
from ding.worker.buffer import DequeBuffer
from ding.utils import BUFFER_REGISTRY


@BUFFER_REGISTRY.register('deque')
class DequeBufferWrapper(object):

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(replay_buffer_size=10000, )

    def __init__(
            self, cfg: EasyDict, tb_logger: Optional[object] = None, exp_name: str = 'default_experiement'
    ) -> None:
        self.buffer = DequeBuffer(size=cfg.replay_buffer_size)

    def sample(self, size: int, train_iter: int):
        output = self.buffer.sample(size=size, ignore_insufficient=True)
        if len(output) > 0:
            return [o.data for o in output]
        else:
            return None

    def push(self, data, cur_collector_envstep: int = -1) -> None:
        # meta = {'train_iter_data_collected': }
        for d in data:
            self.buffer.push(d)
