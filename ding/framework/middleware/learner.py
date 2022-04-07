from typing import TYPE_CHECKING, Callable, List, Tuple, Union, Dict
from easydict import EasyDict
from collections import deque

from ding.framework import task
from ding.data import Buffer
from .functional import trainer, offpolicy_data_fetcher

if TYPE_CHECKING:
    from ding.framework import Context


class OffPolicyLearner:

    def __init__(
            self, cfg: EasyDict, policy, buffer_: Union[Buffer, List[Tuple[Buffer, float]], Dict[str, Buffer]]
    ) -> None:
        self.cfg = cfg
        self._fetcher = task.wrap(offpolicy_data_fetcher(cfg, buffer_))
        self._trainer = task.wrap(trainer(cfg, policy))

    def __call__(self, ctx: "Context") -> None:
        ctx.train_output_queue = deque()
        for _ in range(self.cfg.policy.learn.update_per_collect):
            self._fetcher(ctx)
            if ctx.train_data is None:
                break
            self._trainer(ctx)
            ctx.train_output_queue.append(ctx.train_output)
