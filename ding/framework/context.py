import numpy as np
import dataclasses
from typing import Any


@dataclasses.dataclass
class Context:
    """
    Overview:
        Context is an object that pass contextual data between middlewares, whose life cycle
        is only one training iteration. It is a dict that reflect itself, so you can set
        any properties as you wish.
    """
    _kept_keys: set = dataclasses.field(default_factory=set)
    total_step: int = 0

    def renew(self) -> 'Context':  # noqa
        """
        Overview:
            Renew context from self, add total_step and shift kept properties to the new instance.
        """
        total_step = self.total_step
        ctx = type(self)()
        for key in self._kept_keys:
            if key in dataclasses.asdict(self):
                setattr(ctx, key, getattr(self, key))
        ctx.total_step = total_step + 1
        return ctx

    def keep(self, *keys: str) -> None:
        """
        Overview:
            Keep this key/keys until next iteration.
        """
        for key in keys:
            self._kept_keys.add(key)


# TODO: Restrict data to specific types
@dataclasses.dataclass
class OnlineRLContext(Context):

    # common
    total_step: int = 0
    env_step: int = 0
    env_episode: int = 0
    train_iter: int = 0
    train_data: Any = None
    # collect
    collect_kwargs: Any = dataclasses.field(default_factory=dict)
    trajectories: Any = None
    episodes: Any = None
    trajectory_end_idx: Any = dataclasses.field(default_factory=list)
    # eval
    eval_value: float = -np.inf
    last_eval_iter: int = -1

    def __post_init__(self):
        self.keep('env_step', 'env_episode', 'train_iter', 'last_eval_iter')


@dataclasses.dataclass
class OfflineRLContext(Context):

    # common
    total_step: int = 0
    train_epoch: int = 0
    train_iter: int = 0
    train_data: Any = None
    # eval
    eval_value: float = -np.inf
    last_eval_iter: int = -1

    def __post_init__(self):
        self.keep('train_iter', 'last_eval_iter')
