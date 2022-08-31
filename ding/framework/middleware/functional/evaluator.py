from typing import TYPE_CHECKING, Callable, Any, List, Union
from abc import ABC, abstractmethod
from collections import deque
from ditk import logging
import numpy as np
import torch
import treetensor.torch as ttorch
from easydict import EasyDict
from ding.envs import BaseEnvManager
from ding.policy import Policy
from ding.data import Dataset, DataLoader
from ding.framework import task
from ding.torch_utils import tensor_to_list, to_ndarray
from ding.utils import lists_to_dicts

if TYPE_CHECKING:
    from ding.framework import Context, OnlineRLContext


class IMetric(ABC):

    @abstractmethod
    def eval(self, inputs: Any, label: Any) -> dict:
        raise NotImplementedError

    @abstractmethod
    def reduce_mean(self, inputs: List[Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def gt(self, metric1: Any, metric2: Any) -> bool:
        """
        Overview:
            Whether metric1 is greater than metric2 (>=)

        .. note::
            If metric2 is None, return True
        """
        raise NotImplementedError


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


def interaction_evaluator(cfg: EasyDict, policy: Policy, env: BaseEnvManager) -> Callable:
    """
    Overview:
        The middleware that executes the evaluation.
    Arguments:
        - cfg (:obj:`EasyDict`): Config.
        - policy (:obj:`Policy`): The policy to be evaluated.
        - env (:obj:`BaseEnvManager`): The env for the evaluation.
    """

    env.seed(cfg.seed, dynamic_seed=False)

    def _evaluate(ctx: "OnlineRLContext"):
        """
        Overview:
            - The evaluation will be executed if the task begins and enough train_iter passed \
                since last evaluation.
        Input of ctx:
            - last_eval_iter (:obj:`int`): Last evaluation iteration.
            - train_iter (:obj:`int`): Current train iteration.
        Output of ctx:
            - eval_value (:obj:`float`): The average reward in the current evaluation.
        """

        if ctx.last_eval_iter != -1 and \
           (ctx.train_iter - ctx.last_eval_iter < cfg.policy.eval.evaluator.eval_freq):
            return

        if env.closed:
            env.launch()
        else:
            env.reset()
        policy.reset()
        eval_monitor = VectorEvalMonitor(env.env_num, cfg.env.n_evaluator_episode)

        while not eval_monitor.is_finished():
            obs = ttorch.as_tensor(env.ready_obs).to(dtype=ttorch.float32)
            obs = {i: obs[i] for i in range(obs.shape[0])}  # TBD
            inference_output = policy.forward(obs)
            action = [to_ndarray(v['action']) for v in inference_output.values()]  # TBD
            timesteps = env.step(action)
            for timestep in timesteps:
                env_id = timestep.env_id.item()
                if timestep.done:
                    policy.reset([env_id])
                    reward = timestep.info.final_eval_reward
                    eval_monitor.update_reward(env_id, reward)
        episode_reward = eval_monitor.get_episode_reward()
        eval_reward = np.mean(episode_reward)
        stop_flag = eval_reward >= cfg.env.stop_value and ctx.train_iter > 0
        logging.info(
            'Evaluation: Train Iter({})\tEnv Step({})\tEval Reward({:.3f})'.format(
                ctx.train_iter, ctx.env_step, eval_reward
            )
        )
        ctx.last_eval_iter = ctx.train_iter
        ctx.eval_value = eval_reward

        if stop_flag:
            task.finish = True

    return _evaluate


def metric_evaluator(cfg: EasyDict, policy: Policy, dataset: Dataset, metric: IMetric) -> Callable:
    dataloader = DataLoader(dataset, batch_size=cfg.policy.eval.batch_size)

    def _evaluate(ctx: "Context"):
        # evaluation will be executed if the task begins or enough train_iter after last evaluation
        if ctx.last_eval_iter != -1 and \
           (ctx.train_iter - ctx.last_eval_iter < cfg.policy.eval.evaluator.eval_freq):
            return

        policy.reset()
        eval_output = []

        for batch_idx, batch_data in enumerate(dataloader):
            inputs, label = batch_data
            inference_output = policy.forward(inputs)
            eval_output.append(metric.eval(inference_output, label))
        # TODO reduce avg_eval_output among different gpus
        avg_eval_output = metric.reduce_mean(eval_output)
        stop_flag = metric.gt(avg_eval_output, cfg.env.stop_value) and ctx.train_iter > 0
        logging.info(
            'Evaluation: Train Iter({})\tEnv Step({})\tEval Reward({:.3f})'.format(
                ctx.train_iter, ctx.env_step, avg_eval_output
            )
        )
        ctx.last_eval_iter = ctx.train_iter
        ctx.eval_value = avg_eval_output

        if stop_flag:
            task.finish = True

    return _evaluate


# TODO battle evaluator
