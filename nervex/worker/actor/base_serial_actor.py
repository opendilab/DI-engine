from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import namedtuple
import copy
import numpy as np
from .env_manager import BaseEnvManager
from nervex.utils import build_logger_naive, EasyTimer


class BaseSerialActor(object):

    def __init__(self, cfg: dict) -> None:
        self._default_n_episode = cfg.get('n_episode', None)
        self._default_n_step = cfg.get('n_step', None)
        self._logger, _ = build_logger_naive(path='./log', name='actor')
        self._timer = EasyTimer()
        self._cfg = cfg

    @property
    def env(self) -> BaseEnvManager:
        return self._env

    @env.setter
    def env(self, _env: BaseEnvManager) -> None:
        self._env = _env
        self._env.launch()
        self._env_num = self._env.env_num
        self.reset()

    @property
    def policy(self) -> namedtuple:
        return self._policy

    @policy.setter
    def policy(self, _policy: namedtuple) -> None:
        self._policy = _policy

    def reset(self) -> None:
        self._obs_pool = CachePool('obs', self._env_num)
        self._agent_output_pool = CachePool('agent_output', self._env_num)
        self._transition_buffer = TransitionBuffer(self._env_num)

    def generate_data(self, n_episode: Optional[int] = None, n_step: Optional[int] = None) -> List[Any]:
        assert n_episode is None or n_step is None, "n_episode and n_step can't be not None at the same time"
        if n_episode is not None:
            return self._collect_episode(n_episode)
        elif n_step is not None:
            return self._collect_step(n_step)
        elif self._default_n_episode is not None:
            return self._collect_episode(self._default_n_episode)
        elif self._default_n_step is not None:
            return self._collect_step(self._default_n_step)
        else:
            raise RuntimeError("please indicate specific n_episode or n_step(int value)")

    def _collect_episode(self, n_episode: int) -> List[Any]:
        return self._collect(lambda x, y: x >= n_episode)

    def _collect_step(self, n_step: int) -> List[Any]:
        return self._collect(lambda x, y: y >= n_step)

    def _collect(self, collect_end_fn: Callable) -> List[Any]:
        episode_count = 0
        step_count = 0
        traj_count = 0
        episode_reward = []
        return_data = []
        info = {}
        with self._timer:
            while not collect_end_fn(episode_count, traj_count):
                obs = self._env.next_obs
                self._obs_pool.update(obs)
                env_id, obs = self._policy.data_preprocess(obs)
                agent_output = self._policy.forward(obs)
                agent_output = self._policy.data_postprocess(env_id, agent_output)
                self._agent_output_pool.update(agent_output)
                actions = {env_id: output['action'] for env_id, output in agent_output.items()}
                timesteps = self._env.step(actions)
                for env_id, timestep in timesteps.items():
                    transition = self._policy.process_transition(self._obs_pool[env_id], self._agent_output_pool[env_id], timestep)
                    self._transition_buffer.append(env_id, transition)
                    if timestep.done:
                        # env reset is done by env_manager automatically
                        self._policy.callback_episode_done(env_id)
                        reward = timestep.info['final_eval_reward']
                        episode_reward.append(reward)
                        self._logger.info(
                            "env {} finish episode, final reward: {}, collected episode: {}".format(
                                env_id, reward, episode_count
                            )
                        )
                        episode_count += 1
                    traj = self._policy.get_trajectory(self._transition_buffer, env_id, done=timestep.done)
                    if traj is not None:
                        return_data.extend(traj)
                        traj_count += len(traj)
                        self._logger.info("env {} get new traj, collected traj: {}".format(env_id, traj_count))
                    step_count += 1
        duration = self._timer.value
        info = {
            'episode_count': episode_count,
            'step_count': step_count,
            'traj_count': traj_count,
            'avg_step_per_episode': step_count / (episode_count + 1e-8),
            'avg_traj_per_epsiode': traj_count / (episode_count + 1e-8),
            'avg_time_per_step': duration / (step_count + 1e-8),
            'avg_time_per_traj': duration / (traj_count + 1e-8),
            'avg_time_per_episode': duration / (episode_count + 1e-8),
            'reward_mean': np.mean(episode_reward) if len(episode_reward) > 0 else 0.,
            'reward_std': np.std(episode_reward) if len(episode_reward) > 0 else 0.,
        }
        self._logger.info("collect end:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()])))
        return return_data


class TransitionBuffer(object):

    def __init__(self, env_num: int):
        self._env_num = env_num
        self._buffer = {env_id: [] for env_id in range(env_num)}

    def append(self, env_id: int, transition: dict):
        self._buffer[env_id].append(transition)

    def __getitem__(self, env_id: int) -> List[dict]:
        return self._buffer[env_id]


class CachePool(object):

    def __init__(self, name: str, env_num: int, deepcopy: bool = True):
        self._pool = [None for _ in range(env_num)]
        # TODO(nyz) whether must use deepcopy
        self._deepcopy = deepcopy

    def update(self, data: Dict[int, Any]):
        for i, d in data.items():
            if self._deepcopy:
                self._pool[i] = copy.deepcopy(d)
            else:
                self._pool[i] = d

    def __getitem__(self, idx: int) -> Any:
        return self._pool[idx]
