from typing import Optional, Callable, Tuple, Dict, List
from collections import namedtuple, defaultdict
import numpy as np
import torch

from ...envs import BaseEnvManager
from ...envs import BaseEnvManager

from ding.envs import BaseEnvManager
from ding.torch_utils import to_tensor, to_ndarray, to_item
from ding.utils import build_logger, EasyTimer, SERIAL_EVALUATOR_REGISTRY
from ding.utils import get_world_size, get_rank, broadcast_object_list
from .base_serial_evaluator import ISerialEvaluator, VectorEvalMonitor
from .interaction_serial_evaluator import InteractionSerialEvaluator

class InteractionSerialMetaEvaluator(InteractionSerialEvaluator):
    """
    Overview:
        Interaction serial evaluator class, policy interacts with env. This class evaluator algorithm
        with test environment list.
    Interfaces:
        ``__init__``, reset, reset_policy, reset_env, close, should_eval, eval
    Property:
        env, policy
    """
    config = dict(
        # (int) Evaluate every "eval_freq" training iterations.
        eval_freq=1000,
        render=dict(
            # Tensorboard video render is disabled by default.
            render_freq=-1,
            mode='train_iter',
        ),
        # (str) File path for visualize environment information.
        figure_path=None,
        # test env num
        test_env_num=10,
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
        super().__init__(cfg, env, policy, tb_logger, exp_name, instance_name)
        self.test_env_num = cfg.test_env_num

    def init_params(self, params):
        self.params = params
        self._env.set_all_goals(params)

    def eval(
            self,
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None,
            force_render: bool = False,
            policy_kwargs: Optional[Dict] = {},
            policy_warm_func: namedtuple = None,
            need_reward: bool = False,
    ) -> Tuple[bool, Dict[str, List]]:
        infos = defaultdict(list)
        for i in range(self.test_env_num):
            print('-----------------------------start task ', i)
            self._env.reset_task(i)
            if policy_warm_func is not None:
                policy_warm_func(i)
            info = self.sub_eval(save_ckpt_fn, train_iter, envstep, n_episode, \
                                                  force_render, policy_kwargs, i, need_reward)
            for key, val in info.items():
                if i == 0:
                    info[key] = []
                infos[key].append(val)
        
        meta_infos = defaultdict(list)
        for key, val in infos.items():
            meta_infos[key] = np.array(val).mean()
        
        episode_return = meta_infos['reward_mean']
        meta_infos['train_iter'] = train_iter
        meta_infos['ckpt_name'] = 'iteration_{}.pth.tar'.format(train_iter)

        self._logger.info(self._logger.get_tabulate_vars_hor(meta_infos))
            # self._logger.info(self._logger.get_tabulate_vars(info))
        for k, v in meta_infos.items():
            if k in ['train_iter', 'ckpt_name', 'each_reward']:
                continue
            if not np.isscalar(v):
                continue
            self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
            self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, envstep)

        if episode_return > self._max_episode_return:
            if save_ckpt_fn:
                save_ckpt_fn('ckpt_best.pth.tar')
            self._max_episode_return = episode_return
        
        stop_flag = episode_return >= self._stop_value and train_iter > 0
        if stop_flag:
            self._logger.info(
                "[DI-engine serial pipeline] " + "Current episode_return: {:.4f} is greater than stop_value: {}".
                format(episode_return, self._stop_value) + ", so your RL agent is converged, you can refer to " +
                "'log/evaluator/evaluator_logger.txt' for details."
            )

        return stop_flag, meta_infos
    
    def sub_eval(
            self,
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None,
            force_render: bool = False,
            policy_kwargs: Optional[Dict] = {},
            task_id: int = 0,
            need_reward: bool = False,
    ) -> Tuple[bool, Dict[str, List]]:
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
            - episode_info (:obj:`Dict[str, List]`): Current evaluation episode information.
        '''
        # evaluator only work on rank0
        stop_flag = False
        if get_rank() == 0:
            if n_episode is None:
                n_episode = self._default_n_episode
            assert n_episode is not None, "please indicate eval n_episode"
            envstep_count = 0
            info = {}
            eval_monitor = VectorEvalMonitor(self._env.env_num, n_episode)
            self._env.reset()
            self._policy.reset()

            # force_render overwrite frequency constraint
            render = force_render or self._should_render(envstep, train_iter)

            rewards = None

            with self._timer:
                while not eval_monitor.is_finished():
                    obs = self._env.ready_obs
                    obs = to_tensor(obs, dtype=torch.float32)

                    if need_reward:
                        for id,val in obs.items():
                            if rewards is None:
                                reward = torch.zeros((1))
                            else:
                                reward = torch.tensor(rewards[id], dtype=torch.float32)
                            obs[id] = {'obs':val, 'reward':reward}


                    # update videos
                    if render:
                        eval_monitor.update_video(self._env.ready_imgs)

                    if self._policy_cfg.type == 'dreamer_command':
                        policy_output = self._policy.forward(
                            obs, **policy_kwargs, reset=self._resets, state=self._states
                        )
                        #self._states = {env_id: output['state'] for env_id, output in policy_output.items()}
                        self._states = [output['state'] for output in policy_output.values()]
                    else:
                        policy_output = self._policy.forward(obs, **policy_kwargs)
                    actions = {i: a['action'] for i, a in policy_output.items()}
                    actions = to_ndarray(actions)
                    timesteps = self._env.step(actions)
                    timesteps = to_tensor(timesteps, dtype=torch.float32)
                    rewards = []
                    for env_id, t in timesteps.items():
                        rewards.append(t.reward)
                        if t.info.get('abnormal', False):
                            # If there is an abnormal timestep, reset all the related variables(including this env).
                            self._policy.reset([env_id])
                            continue
                        if self._policy_cfg.type == 'dreamer_command':
                            self._resets[env_id] = t.done
                        if t.done:
                            # Env reset is done by env_manager automatically.
                            if 'figure_path' in self._cfg and self._cfg.figure_path is not None:
                                self._env.enable_save_figure(env_id, self._cfg.figure_path)
                            self._policy.reset([env_id])
                            reward = t.info['eval_episode_return']
                            saved_info = {'eval_episode_return': t.info['eval_episode_return']}
                            if 'episode_info' in t.info:
                                saved_info.update(t.info['episode_info'])
                            eval_monitor.update_info(env_id, saved_info)
                            eval_monitor.update_reward(env_id, reward)
                            self._logger.info(
                                "[EVALUATOR]env {} finish task {} episode, final reward: {:.4f}, current episode: {}".format(
                                    env_id, task_id, eval_monitor.get_latest_reward(env_id), eval_monitor.get_current_episode()
                                )
                            )
                        envstep_count += 1
            duration = self._timer.value
            episode_return = eval_monitor.get_episode_return()
            info = {
                'episode_count': n_episode,
                'envstep_count': envstep_count,
                'avg_envstep_per_episode': envstep_count / n_episode,
                'evaluate_time': duration,
                'avg_envstep_per_sec': envstep_count / duration,
                'avg_time_per_episode': n_episode / duration,
                'reward_mean': np.mean(episode_return),
                'reward_std': np.std(episode_return),
                'reward_max': np.max(episode_return),
                'reward_min': np.min(episode_return),
                # 'each_reward': episode_return,
            }
            episode_info = eval_monitor.get_episode_info()
            if episode_info is not None:
                info.update(episode_info)

            if render:
                video_title = '{}_{}/'.format(self._instance_name, self._render.mode)
                videos = eval_monitor.get_video()
                render_iter = envstep if self._render.mode == 'envstep' else train_iter
                from ding.utils import fps
                self._tb_logger.add_video(video_title, videos, render_iter, fps(self._env))     

        return info
