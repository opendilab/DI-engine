import numpy as np
import dataclasses
import treetensor.torch as ttorch
from typing import Union, Dict, List


@dataclasses.dataclass
class Context:
    """
    Overview:
        Context is an object that pass contextual data between middlewares, whose life cycle
        is only one training iteration. It is a dict that reflect itself, so you can set
        any properties as you wish.
        Note that the initial value of the property must be equal to False.
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
            if self.has_attr(key):
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

    def has_attr(self, key):
        return hasattr(self, key)


# TODO: Restrict data to specific types
@dataclasses.dataclass
class OnlineRLContext(Context):

    # common
    total_step: int = 0
    env_step: int = 0
    env_episode: int = 0
    train_iter: int = 0
    train_data: Union[Dict, List] = None
    train_output: Union[Dict, List[Dict]] = None
    # collect
    collect_kwargs: Dict = dataclasses.field(default_factory=dict)
    obs: ttorch.Tensor = None
    action: List = None
    inference_output: Dict[int, Dict] = None
    trajectories: List = None
    episodes: List = None
    trajectory_end_idx: List = dataclasses.field(default_factory=list)
    action: Dict = None
    inference_output: Dict = None
    # eval
    eval_value: float = -np.inf
    last_eval_iter: int = -1
    last_eval_value: int = -np.inf
    eval_output: List = dataclasses.field(default_factory=dict)
    # wandb
    wandb_url: str = ""

    def __post_init__(self):
        # This method is called just after __init__ method. Here, concretely speaking,
        # this method is called just after the object initialize its fields.
        # We use this method here to keep the fields needed for each iteration.
        self.keep('env_step', 'env_episode', 'train_iter', 'last_eval_iter', 'last_eval_value', 'wandb_url')


@dataclasses.dataclass
class OfflineRLContext(Context):

    # common
    total_step: int = 0
    trained_env_step: int = 0
    train_epoch: int = 0
    train_iter: int = 0
    train_data: Union[Dict, List] = None
    train_output: Union[Dict, List[Dict]] = None
    # eval
    eval_value: float = -np.inf
    last_eval_iter: int = -1
    last_eval_value: int = -np.inf
    eval_output: List = dataclasses.field(default_factory=dict)
    # wandb
    wandb_url: str = ""

    def __post_init__(self):
        # This method is called just after __init__ method. Here, concretely speaking,
        # this method is called just after the object initialize its fields.
        # We use this method here to keep the fields needed for each iteration.
        self.keep('trained_env_step', 'train_iter', 'last_eval_iter', 'last_eval_value', 'wandb_url')
