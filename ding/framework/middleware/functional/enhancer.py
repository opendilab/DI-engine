from typing import TYPE_CHECKING, Callable
from easydict import EasyDict
import logging
from ding.policy import Policy
if TYPE_CHECKING:
    from ding.framework import OnlineRLContext
    from ding.reward_model import BaseRewardModel, HerRewardModel
    from ding.data import Buffer


def reward_estimator(cfg: EasyDict, reward_model: "BaseRewardModel") -> Callable:

    def _enhance(ctx: "OnlineRLContext"):
        reward_model.estimate(ctx.train_data)  # inplace modification

    return _enhance


def her_data_enhancer(cfg: EasyDict, buffer_: "Buffer", her_reward_model: "HerRewardModel") -> Callable:

    def _fetch_and_enhance(ctx: "OnlineRLContext"):
        if her_reward_model.episode_size is None:
            size = cfg.policy.learn.batch_size
        else:
            size = her_reward_model.episode_size
        try:
            buffered_episode = buffer_.sample(size)
            train_episode = [d.data for d in buffered_episode]
        except (ValueError, AssertionError):
            # You can modify data collect config to avoid this warning, e.g. increasing n_sample, n_episode.
            logging.warning(
                "Replay buffer's data is not enough to support training, so skip this training for waiting more data."
            )
            ctx.train_data = None
            return

        her_episode = sum([her_reward_model.estimate(e) for e in train_episode], [])
        ctx.train_data = sum(her_episode, [])

    return _fetch_and_enhance


# TODO nstep reward
# TODO MBPO
# TODO SIL
# TODO TD3 VAE
