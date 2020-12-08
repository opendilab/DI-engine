from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import namedtuple
import copy
import numpy as np
from nervex.worker import BaseEnvManager
from nervex.utils import build_logger, EasyTimer


class BaseSerialActor(object):

    def __init__(self, cfg: dict) -> None:
        self._default_n_episode = cfg.get('n_episode', None)
        self._default_n_step = cfg.get('n_step', None)
        self._logger = build_logger()
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
        episode_reward = {}
        return_data = []
        info = {}
        with self._timer:
            while not collect_end_fn(episode_count, step_count):
                obs = self._env.next_obs
                self._obs_pool.update(obs)
                obs, env_id = self._policy.data_preprocess(obs)
                agent_output = self._policy.forward(obs)
                agent_output = self._policy.data_postprocess(env_id, agent_output)
                self._agent_output_pool.update(agent_output)
                action = {i: a['action'] for i, a in agent_output.items()}
                timestep = self._env.step(action)
                for i, t in timestep.items():
                    transition = self._policy.process_transition(self._obs_pool[i], self._agent_output_pool[i], t)
                    self._transition_buffer.append(i, transition)
                    if t.done:
                        # env reset is done by env_manager automatically
                        self._policy.callback_episode_done(i)
                        reward = t.info[i]['final_eval_reward']
                        episode_reward.append(reward)
                        self._logger.info(
                            "env {} finish episode, final reward: {}, current episode: {}".format(
                                i, reward, episode_count
                            )
                        )
                        episode_count += 1
                    traj = self._policy.get_trajectory(self._transition_buffer, i, done=t.done)
                    if traj is not None:
                        return_data.extend(traj)
                        traj_count += len(traj)
                        self._logger.info("env {} get new traj, current traj: {}".format(i, step_count))
                    step_count += 1
        duration = self._timer.value
        info = {
            'episode_count': episode_count,
            'step_count': step_count,
            'traj_count': traj_count,
            'avg_step_per_episode': step_count / episode_count,
            'avg_traj_per_epsiode': traj_count / episode_count,
            'avg_time_per_step': duration / step_count,
            'avg_time_per_traj': duration / traj_count,
            'avg_time_per_episode': duration / episode_count,
            'reward_mean': np.mean(episode_reward),
            'reward_std': np.std(episode_reward)
        }
        self._logger.info("collect end:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()])))
        return return_data


class TransitionBuffer(object):
    pass


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
