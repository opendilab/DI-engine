from typing import TYPE_CHECKING, Callable
from easydict import EasyDict
from ditk import logging
import torch
from ding.policy import Policy
if TYPE_CHECKING:
    from ding.framework import OnlineRLContext
    from ding.reward_model import BaseRewardModel, HerRewardModel
    from ding.data import Buffer


def reward_estimator(cfg: EasyDict, reward_model: "BaseRewardModel") -> Callable:
    """
    Overview:
        Estimate the reward of `train_data` using `reward_model`.
    Arguments:
        - cfg (:obj:`EasyDict`): Config.
        - reward_model (:obj:`BaseRewardModel`): Reward model.
    """

    def _enhance(ctx: "OnlineRLContext"):
        """
        Input of ctx:
            - train_data (:obj:`List`): The list of data used for estimation.
        """
        reward_model.estimate(ctx.train_data)  # inplace modification

    return _enhance


def her_data_enhancer(cfg: EasyDict, buffer_: "Buffer", her_reward_model: "HerRewardModel") -> Callable:
    """
    Overview:
        Fetch a batch of data/episode from `buffer_`, \
        then use `her_reward_model` to get HER processed episodes from original episodes.
    Arguments:
        - cfg (:obj:`EasyDict`): Config which should contain the following keys \
            if her_reward_model.episode_size is None: `cfg.policy.learn.batch_size`.
        - buffer\_ (:obj:`Buffer`): Buffer to sample data from.
        - her_reward_model (:obj:`HerRewardModel`): Hindsight Experience Replay (HER) model \
            which is used to process episodes.
    """

    def _fetch_and_enhance(ctx: "OnlineRLContext"):
        """
        Output of ctx:
            - train_data (:obj:`List[treetensor.torch.Tensor]`): The HER processed episodes.
        """
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


def nstep_reward_enhancer(cfg: EasyDict) -> Callable:

    def _enhance(ctx: "OnlineRLContext"):
        nstep = cfg.policy.nstep
        gamma = cfg.policy.discount_factor
        L = len(ctx.trajectories)
        reward_template = ctx.trajectories[0].reward
        nstep_rewards = []
        value_gamma = []
        for i in range(L):
            valid = min(nstep, L - i)
            for j in range(1, valid):
                if ctx.trajectories[j + i].done:
                    valid = j
                    break
            value_gamma.append(torch.FloatTensor([gamma ** valid]))
            nstep_reward = [ctx.trajectories[j].reward for j in range(i, i + valid)]
            if nstep > valid:
                nstep_reward.extend([torch.zeros_like(reward_template) for j in range(nstep - valid)])
            nstep_reward = torch.cat(nstep_reward)  # (nstep, )
            nstep_rewards.append(nstep_reward)
        for i in range(L):
            ctx.trajectories[i].reward = nstep_rewards[i]
            ctx.trajectories[i].value_gamma = value_gamma[i]

    return _enhance


# TODO MBPO
# TODO SIL
# TODO TD3 VAE
