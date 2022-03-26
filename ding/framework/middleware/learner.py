from typing import TYPE_CHECKING, Callable
from easydict import EasyDict

from ding.framework import task
from ding.data import Buffer
from .functional import trainer, offpolicy_data_fetcher

if TYPE_CHECKING:
    from ding.framework import Context


class OffPolicyLearner:

    def __init__(self, cfg: EasyDict, policy, buffer_: Buffer) -> None:
        self.cfg = cfg
        self._fetcher = offpolicy_data_fetcher(cfg, buffer_)
        self._trainer = trainer(cfg, policy)

    def __call__(self, ctx: "Context") -> None:
        for _ in range(self.cfg.policy.learn.update_per_collect):
            self._fetcher(ctx)
            if ctx.train_data is None:
                break
            self._trainer(ctx)
