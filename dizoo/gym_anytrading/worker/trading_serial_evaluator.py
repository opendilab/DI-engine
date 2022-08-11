from typing import Any, Optional, Callable, Tuple
from collections import deque, namedtuple
from easydict import EasyDict
import torch
import numpy as np

from ding.envs import BaseEnvManager
from ding.worker import VectorEvalMonitor, InteractionSerialEvaluator
from ding.torch_utils import to_tensor, to_ndarray
from ding.utils import SERIAL_EVALUATOR_REGISTRY, import_module


@SERIAL_EVALUATOR_REGISTRY.register('trading_interaction')
class TradingSerialEvaluator(InteractionSerialEvaluator):
    """
    Overview:
        Trading interaction serial evaluator class, policy interacts with anytrading env.
    Interfaces:
        __init__, reset, reset_policy, reset_env, close, should_eval, eval
    Property:
        env, policy
    """
    config = dict(
        # Evaluate every "eval_freq" training iterations.
        eval_freq=1000,
        render=dict(
            # tensorboard video render is disabled by default
            render_freq=-1,
            mode='train_iter',
        ),
        type='trading_interaction',
    )
    def __init__(
            self,
            cfg: dict,
            env: BaseEnvManager = None,
            policy: namedtuple = None,
            tb_logger: 'SummaryWriter' = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'evaluator',
    ) -> None:
        """
        Overview:
            Init method. Just init super class.
        Arguments:
            - cfg (:obj:`EasyDict`): Configuration EasyDict.
        """
        super().__init__(cfg, env, policy, tb_logger, exp_name, instance_name)

    def eval(
            self,
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None,
            force_render: bool = False,
    ) -> Tuple[bool, dict]:
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
            - return_info (:obj:`dict`): Current evaluation return information.
        '''

        if n_episode is None:
            n_episode = self._default_n_episode
        assert n_episode is not None, "please indicate eval n_episode"
        envstep_count = 0
        info = {}
        return_info = []
        eval_monitor = TradingEvalMonitor(self._env.env_num, n_episode)
        self._env.reset()
        self._policy.reset()

        # force_render overwrite frequency constraint
        render = force_render or self._should_render(envstep, train_iter)

        with self._timer:
            while not eval_monitor.is_finished():
                obs = self._env.ready_obs
                obs = to_tensor(obs, dtype=torch.float32)

                # update videos
                if render:
                    eval_monitor.update_video(self._env.ready_imgs)

                policy_output = self._policy.forward(obs)
                actions = {i: a['action'] for i, a in policy_output.items()}
                actions = to_ndarray(actions)
                timesteps = self._env.step(actions)
                timesteps = to_tensor(timesteps, dtype=torch.float32)
                for env_id, t in timesteps.items():
                    if t.info.get('abnormal', False):
                        # If there is an abnormal timestep, reset all the related variables(including this env).
                        self._policy.reset([env_id])
                        continue
                    if t.done:
                        # Env reset is done by env_manager automatically.
                        self._policy.reset([env_id])
                        reward = t.info['final_eval_reward']
                        if 'episode_info' in t.info:
                            eval_monitor.update_info(env_id, t.info['episode_info'])
                        eval_monitor.update_reward(env_id, reward)
                        return_info.append(t.info)

                        #========== only used by anytrading =======
                        if 'max_possible_profit' in t.info:
                            max_profit = t.info['max_possible_profit']
                            eval_monitor.update_max_profit(env_id, max_profit)
                        #==========================================

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

        #========== only used by anytrading =======
        max_possible_profit = eval_monitor.get_max_episode_profit()
        info_anytrading = {
            'max_possible_profit_max': np.max(max_possible_profit),
            'max_possible_profit_mean': np.mean(max_possible_profit),
            'max_possible_profit_min': np.min(max_possible_profit),
        }
        for k, v in info_anytrading.items():
            if not np.isscalar(v):
                continue
            self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
            self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, envstep)
        #==========================================

        if render:
            video_title = '{}_{}/'.format(self._instance_name, self._render.mode)
            videos = eval_monitor.get_video()
            render_iter = envstep if self._render.mode == 'envstep' else train_iter
            from ding.utils import fps
            self._tb_logger.add_video(video_title, videos, render_iter, fps(self._env))

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


class TradingEvalMonitor(VectorEvalMonitor):
    """
    Overview:
        Inherit VectorEvalMonitor for trading env.
        Add func update_max_profit and get_max_episode_profit in order to log the max_profit for every episode.
    Interfaces:
        Besides (__init__, is_finished, update_info, update_reward, get_episode_reward,\
            get_latest_reward, get_current_episode, get_episode_info), there are\
                (update_max_profit, get_max_episode_profit).
    """

    def __init__(self, env_num: int, n_episode: int) -> None:
        super().__init__(env_num, n_episode)

        self._each_env_episode = [n_episode // env_num for _ in range(env_num)]
        self._max_possible_profit = {
            env_id: deque(maxlen=maxlen)
            for env_id, maxlen in enumerate(self._each_env_episode)
        }

    def update_max_profit(self, env_id: int, max_profit: Any) -> None:
        """
        Overview:
            Update the max profit indicated by env_id.
        Arguments:
            - env_id: (:obj:`int`): the id of the environment we need to update the max profit
            - max_profit: (:obj:`Any`): the profit we need to update
        """
        if isinstance(max_profit, torch.Tensor):
            max_profit = max_profit.item()
        self._max_possible_profit[env_id].append(max_profit)

    def get_max_episode_profit(self) -> list:
        return sum([list(v) for v in self._max_possible_profit.values()], [])
