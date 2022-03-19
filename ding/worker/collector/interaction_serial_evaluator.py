from typing import Optional, Callable, Tuple
from collections import namedtuple
import numpy as np
import torch

from ding.envs import BaseEnvManager
from ding.torch_utils import to_tensor, to_ndarray
from ding.utils import build_logger, EasyTimer, SERIAL_EVALUATOR_REGISTRY
from .base_serial_evaluator import ISerialEvaluator, VectorEvalMonitor


@SERIAL_EVALUATOR_REGISTRY.register('interaction')
class InteractionSerialEvaluator(ISerialEvaluator):
    """
    Overview:
        Interaction serial evaluator class, policy interacts with env.
    Interfaces:
        __init__, reset, reset_policy, reset_env, close, should_eval, eval
    Property:
        env, policy
    """

    config = dict(
        # Evaluate every "eval_freq" training iterations.
        eval_freq=1000,
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
            Init method. Load config and use ``self._cfg`` setting to build common serial evaluator components,
            e.g. logger helper, timer.
        Arguments:
            - cfg (:obj:`EasyDict`): Configuration EasyDict.
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

    def reset_policy(self, _policy: Optional[namedtuple] = None) -> None:
        """
        Overview:
            Reset evaluator's policy. In some case, we need evaluator work in this same environment but use\
                different policy. We can use reset_policy to reset the policy.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the evaluator with the new passed in policy.
        Arguments:
            - policy (:obj:`Optional[namedtuple]`): the api namedtuple of eval_mode policy
        """
        assert hasattr(self, '_env'), "please set env first"
        if _policy is not None:
            self._policy = _policy
        self._policy.reset()

    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset evaluator's policy and environment. Use new policy and environment to collect data.
            If _env is None, reset the old environment.
            If _env is not None, replace the old environment in the evaluator with the new passed in \
                environment and launch.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the evaluator with the new passed in policy.
        Arguments:
            - policy (:obj:`Optional[namedtuple]`): the api namedtuple of eval_mode policy
            - env (:obj:`Optional[BaseEnvManager]`): instance of the subclass of vectorized \
                env_manager(BaseEnvManager)
        """
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)
        self._max_eval_reward = float("-inf")
        self._last_eval_iter = -1
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
        if train_iter == self._last_eval_iter:
            return False
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
        eval_monitor = VectorEvalMonitor(self._env.env_num, n_episode)
        self._env.reset()
        self._policy.reset()

        with self._timer:
            while not eval_monitor.is_finished():
                obs = self._env.ready_obs
                obs = to_tensor(obs, dtype=torch.float32)
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
