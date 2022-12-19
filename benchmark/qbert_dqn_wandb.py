from typing import TYPE_CHECKING, Callable, Any, List, Union
import sys
from copy import deepcopy
from collections import deque
import gym
import torch
import treetensor.torch as ttorch
import numpy as np

from ditk import logging
from ding.model import DQN
from ding.policy import DQNPolicy
from ding.policy import Policy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2, BaseEnvManager, SubprocessEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OfflineRLContext
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, data_pusher, \
    eps_greedy_handler, CkptSaver, termination_checker, nstep_reward_enhancer, wandb_online_logger, interaction_evaluator
from ding.utils import set_pkg_seed
from ding.utils import lists_to_dicts
from ding.utils.log_helper import build_logger
from ding.torch_utils import tensor_to_list, to_ndarray
from dizoo.atari.envs.atari_env import AtariEnv
import wandb
from easydict import EasyDict

from qbert_dqn_config import main_config, create_config


class VectorEvalMonitor(object):
    """
    Overview:
        In some cases,  different environment in evaluator may collect different length episode. For example, \
            suppose we want to collect 12 episodes in evaluator but only have 5 environments, if we didn’t do \
            any thing, it is likely that we will get more short episodes than long episodes. As a result, \
            our average reward will have a bias and may not be accurate. we use VectorEvalMonitor to solve the problem.
    Interfaces:
        __init__, is_finished, update_info, update_reward, get_episode_reward, get_latest_reward, get_current_episode,\
            get_episode_info
    """

    def __init__(self, env_num: int, n_episode: int) -> None:
        """
        Overview:
            Init method. According to the number of episodes and the number of environments, determine how many \
                episodes need to be opened for each environment, and initialize the reward, info and other \
                information
        Arguments:
            - env_num (:obj:`int`): the number of episodes need to be open
            - n_episode (:obj:`int`): the number of environments
        """
        assert n_episode >= env_num, "n_episode < env_num, please decrease the number of eval env"
        self._env_num = env_num
        self._n_episode = n_episode
        each_env_episode = [n_episode // env_num for _ in range(env_num)]
        for i in range(n_episode % env_num):
            each_env_episode[i] += 1
        self._reward = {env_id: deque(maxlen=maxlen) for env_id, maxlen in enumerate(each_env_episode)}
        self._info = {env_id: deque(maxlen=maxlen) for env_id, maxlen in enumerate(each_env_episode)}

    def is_finished(self) -> bool:
        """
        Overview:
            Determine whether the evaluator has completed the work.
        Return:
            - result: (:obj:`bool`): whether the evaluator has completed the work
        """
        return all([len(v) == v.maxlen for v in self._reward.values()])

    def update_info(self, env_id: int, info: Any) -> None:
        """
        Overview:
            Update the information of the environment indicated by env_id.
        Arguments:
            - env_id: (:obj:`int`): the id of the environment we need to update information
            - info: (:obj:`Any`): the information we need to update
        """
        info = tensor_to_list(info)
        self._info[env_id].append(info)

    def update_reward(self, env_id: Union[int, np.ndarray], reward: Any) -> None:
        """
        Overview:
            Update the reward indicated by env_id.
        Arguments:
            - env_id: (:obj:`int`): the id of the environment we need to update the reward
            - reward: (:obj:`Any`): the reward we need to update
        """
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        if isinstance(env_id, np.ndarray):
            env_id = env_id.item()
        self._reward[env_id].append(reward)

    def get_episode_reward(self) -> list:
        """
        Overview:
            Get the total reward of one episode.
        """
        return sum([list(v) for v in self._reward.values()], [])  # sum(iterable, start)

    def get_latest_reward(self, env_id: int) -> int:
        """
        Overview:
            Get the latest reward of a certain environment.
        Arguments:
            - env_id: (:obj:`int`): the id of the environment we need to get reward.
        """
        return self._reward[env_id][-1]

    def get_current_episode(self) -> int:
        """
        Overview:
            Get the current episode. We can know which episode our evaluator is executing now.
        """
        return sum([len(v) for v in self._reward.values()])

    def get_episode_info(self) -> dict:
        """
        Overview:
            Get all episode information, such as total reward of one episode.
        """
        if len(self._info[0]) == 0:
            return None
        else:
            total_info = sum([list(v) for v in self._info.values()], [])
            total_info = lists_to_dicts(total_info)
            new_dict = {}
            for k in total_info.keys():
                if np.isscalar(total_info[k][0]):
                    new_dict[k + '_mean'] = np.mean(total_info[k])
            total_info.update(new_dict)
            return total_info


num_seed = 1


def main(seed):
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    cfg.env.seed = seed

    for i in range(num_seed):
        wandb.init(
            # Set the project where this run will be logged
            project='zjow-QbertNoFrameskip-v4-3',
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"dqn",
            # Track hyperparameters and run metadata
            config=cfg
        )
        logger_, tb_logger = build_logger(path='./log/qbert_dqn/seed' + str(seed), need_tb=True)
        with task.start(async_mode=False, ctx=OnlineRLContext()):
            collector_cfg = deepcopy(cfg.env)
            collector_cfg.is_train = True
            evaluator_cfg = deepcopy(cfg.env)
            evaluator_cfg.is_train = False
            collector_env = SubprocessEnvManagerV2(
                env_fn=[lambda: AtariEnv(collector_cfg) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
            )
            evaluator_env = SubprocessEnvManagerV2(
                env_fn=[lambda: AtariEnv(evaluator_cfg) for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager
            )

            # collector_env.enable_save_replay(replay_path='./lunarlander_video_train')
            evaluator_env.enable_save_replay(replay_path=cfg.policy.logger.record_path)

            set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

            model = DQN(**cfg.policy.model)
            buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
            policy = DQNPolicy(cfg.policy, model=model)

            task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
            task.use(eps_greedy_handler(cfg))
            task.use(StepCollector(cfg, policy.collect_mode, collector_env))
            task.use(nstep_reward_enhancer(cfg))
            task.use(data_pusher(cfg, buffer_))
            task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
            task.use(wandb_online_logger(cfg.policy.logger, evaluator_env, model))
            #task.use(CkptSaver(cfg, policy, train_freq=100))
            task.use(termination_checker(max_env_step=int(10e6)))
            #task.use(_add_scalar)
            task.run()


if __name__ == "__main__":
    main(0)