from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import namedtuple, deque
from easydict import EasyDict
from functools import reduce
import copy
import numpy as np
import torch

from ding.utils import build_logger, EasyTimer, deep_merge_dicts, lists_to_dicts, dicts_to_lists, \
    SERIAL_EVALUATOR_REGISTRY
from ding.envs import BaseEnvManager
from ding.torch_utils import to_tensor, to_ndarray, tensor_to_list
from .base_serial_evaluator import ISerialEvaluator


@SERIAL_EVALUATOR_REGISTRY.register('battle_interaction')
class BattleInteractionSerialEvaluator(ISerialEvaluator):
    """
    Overview:
        Multiple player battle evaluator class.
    Interfaces:
        __init__, reset, reset_policy, reset_env, close, should_eval, eval
    Property:
        env, policy
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        """
        Overview:
            Get evaluator's default config. We merge evaluator's default config with other default configs\
                and user's config to get the final config.
        Return:
            cfg: (:obj:`EasyDict`): evaluator's default config
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        # Evaluate every "eval_freq" training iterations.
        eval_freq=50,
    )

    def __init__(
            self,
            cfg: dict,
            env: BaseEnvManager = None,
            policy: List[namedtuple] = None,
            tb_logger: 'SummaryWriter' = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'evaluator',
    ) -> None:
        """
        Overview:
            Init method. Load config and use ``self._cfg`` setting to build common serial evaluator components,
            e.g. logger helper, timer.
            Policy is not initialized here, but set afterwards through policy setter.
        Arguments:
            - cfg (:obj:`EasyDict`)
        """
        self._cfg = cfg
        self._exp_name = exp_name
        self._instance_name = instance_name
        if tb_logger is not None:
            self._logger, _ = build_logger(
                path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name, need_tb=False
            )
            self._tb_logger = tb_logger
        else:
            self._logger, self._tb_logger = build_logger(
                path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name
            )
        self.reset(policy, env)

        self._timer = EasyTimer()
        self._default_n_episode = cfg.n_episode
        self._stop_value = cfg.stop_value

    def reset_env(self, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset evaluator's environment. In some case, we need evaluator use the same policy in different \
                environments. We can use reset_env to reset the environment.
            If _env is None, reset the old environment.
            If _env is not None, replace the old environment in the evaluator with the \
                new passed in environment and launch.
        Arguments:
            - env (:obj:`Optional[BaseEnvManager]`): instance of the subclass of vectorized \
                env_manager(BaseEnvManager)
        """
        if _env is not None:
            self._env = _env
            self._env.launch()
            self._env_num = self._env.env_num
        else:
            self._env.reset()

    def reset_policy(self, _policy: Optional[List[namedtuple]] = None) -> None:
        """
        Overview:
            Reset evaluator's policy. In some case, we need evaluator work in this same environment but use\
                different policy. We can use reset_policy to reset the policy.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the evaluator with the new passed in policy.
        Arguments:
            - policy (:obj:`Optional[List[namedtuple]]`): the api namedtuple of eval_mode policy
        """
        assert hasattr(self, '_env'), "please set env first"
        if _policy is not None:
            assert len(_policy) > 1, "battle evaluator needs more than 1 policy, but found {}".format(len(_policy))
            self._policy = _policy
            self._policy_num = len(self._policy)
        for p in self._policy:
            p.reset()

    def reset(self, _policy: Optional[List[namedtuple]] = None, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset evaluator's policy and environment. Use new policy and environment to collect data.
            If _env is None, reset the old environment.
            If _env is not None, replace the old environment in the evaluator with the new passed in \
                environment and launch.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the evaluator with the new passed in policy.
        Arguments:
            - policy (:obj:`Optional[List[namedtuple]]`): the api namedtuple of eval_mode policy
            - env (:obj:`Optional[BaseEnvManager]`): instance of the subclass of vectorized \
                env_manager(BaseEnvManager)
        """
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)
        self._max_eval_reward = float("-inf")
        self._last_eval_iter = 0
        self._end_flag = False

    def close(self) -> None:
        """
        Overview:
            Close the evaluator. If end_flag is False, close the environment, flush the tb_logger\
                and close the tb_logger.
        """
        if self._end_flag:
            return
        self._end_flag = True
        self._env.close()
        self._tb_logger.flush()
        self._tb_logger.close()

    def __del__(self):
        """
        Overview:
            Execute the close command and close the evaluator. __del__ is automatically called \
                to destroy the evaluator instance when the evaluator finishes its work
        """
        self.close()

    def should_eval(self, train_iter: int) -> bool:
        """
        Overview:
            Determine whether you need to start the evaluation mode, if the number of training has reached\
                the maximum number of times to start the evaluator, return True
        """
        if (train_iter - self._last_eval_iter) < self._cfg.eval_freq and train_iter != 0:
            return False
        self._last_eval_iter = train_iter
        return True

    def eval(
            self,
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None
    ) -> Tuple[bool, List[dict]]:
        '''
        Overview:
            Evaluate policy and store the best policy based on whether it reaches the highest historical reward.
        Arguments:
            - save_ckpt_fn (:obj:`Callable`): Saving ckpt function, which will be triggered by getting the best reward.
            - train_iter (:obj:`int`): Current training iteration.
            - envstep (:obj:`int`): Current env interaction step.
            - n_episode (:obj:`int`): Number of evaluation episodes.
        Returns:
            - stop_flag (:obj:`bool`): Whether this training program can be ended.
            - return_info (:obj:`list`): Environment information of each finished episode.
        '''
        if n_episode is None:
            n_episode = self._default_n_episode
        assert n_episode is not None, "please indicate eval n_episode"
        envstep_count = 0
        info = {}
        return_info = [[] for _ in range(self._policy_num)]
        eval_monitor = VectorEvalMonitor(self._env.env_num, n_episode)
        self._env.reset()
        for p in self._policy:
            p.reset()

        with self._timer:
            while not eval_monitor.is_finished():
                obs = self._env.ready_obs
                ready_env_id = obs.keys()
                obs = to_tensor(obs, dtype=torch.float32)
                obs = dicts_to_lists(obs)
                policy_output = [p.forward(obs[i]) for i, p in enumerate(self._policy)]
                actions = {}
                for env_id in ready_env_id:
                    actions[env_id] = []
                    for output in policy_output:
                        actions[env_id].append(output[env_id]['action'])
                actions = to_ndarray(actions)
                timesteps = self._env.step(actions)
                timesteps = to_tensor(timesteps, dtype=torch.float32)
                for env_id, t in timesteps.items():
                    if t.done:
                        # Env reset is done by env_manager automatically.
                        for p in self._policy:
                            p.reset([env_id])
                        # policy0 is regarded as main policy default
                        reward = t.info[0]['final_eval_reward']
                        if 'episode_info' in t.info[0]:
                            eval_monitor.update_info(env_id, t.info[0]['episode_info'])
                        eval_monitor.update_reward(env_id, reward)
                        for policy_id in range(self._policy_num):
                            return_info[policy_id].append(t.info[policy_id])
                        self._logger.info(
                            "[EVALUATOR]env {} finish episode, final reward: {}, current episode: {}".format(
                                env_id, eval_monitor.get_latest_reward(env_id), eval_monitor.get_current_episode()
                            )
                        )
                    envstep_count += 1
        duration = self._timer.value
        episode_reward = eval_monitor.get_episode_reward()
        info = {
            'train_iter': train_iter,
            'ckpt_name': 'iteration_{}.pth.tar'.format(train_iter),
            'episode_count': n_episode,
            'envstep_count': envstep_count,
            'avg_envstep_per_episode': envstep_count / n_episode,
            'evaluate_time': duration,
            'avg_envstep_per_sec': envstep_count / duration,
            'avg_time_per_episode': n_episode / duration,
            'reward_mean': np.mean(episode_reward),
            'reward_std': np.std(episode_reward),
            'reward_max': np.max(episode_reward),
            'reward_min': np.min(episode_reward),
            # 'each_reward': episode_reward,
        }
        episode_info = eval_monitor.get_episode_info()
        if episode_info is not None:
            info.update(episode_info)
        self._logger.info(self._logger.get_tabulate_vars_hor(info))
        # self._logger.info(self._logger.get_tabulate_vars(info))
        for k, v in info.items():
            if k in ['train_iter', 'ckpt_name', 'each_reward']:
                continue
            if not np.isscalar(v):
                continue
            self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
            self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, envstep)
        eval_reward = np.mean(episode_reward)
        if eval_reward > self._max_eval_reward:
            if save_ckpt_fn:
                save_ckpt_fn('ckpt_best.pth.tar')
            self._max_eval_reward = eval_reward
        stop_flag = eval_reward >= self._stop_value and train_iter > 0
        if stop_flag:
            self._logger.info(
                "[DI-engine serial pipeline] " +
                "Current eval_reward: {} is greater than stop_value: {}".format(eval_reward, self._stop_value) +
                ", so your RL agent is converged, you can refer to 'log/evaluator/evaluator_logger.txt' for details."
            )
        return stop_flag, return_info


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

    def update_reward(self, env_id: int, reward: Any) -> None:
        """
        Overview:
            Update the reward indicated by env_id.
        Arguments:
            - env_id: (:obj:`int`): the id of the environment we need to update the reward
            - reward: (:obj:`Any`): the reward we need to update
        """
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
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
