from typing import TYPE_CHECKING, Callable
from easydict import EasyDict
from ding.policy import Policy
from ding.reward_model import BaseRewardModel
if TYPE_CHECKING:
    from ding.framework import Context


def reward_estimator(cfg: EasyDict, reward_model: BaseRewardModel) -> Callable:

    def _enhance(ctx: "Context"):
        reward_model.estimate(ctx.train_data)  # inplace modification

    return _enhance


# TODO nstep reward
# TODO MBPO
# TODO SIL
# TODO TD3 VAE
