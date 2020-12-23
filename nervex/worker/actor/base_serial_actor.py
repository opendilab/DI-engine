from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import namedtuple, deque
import copy
import numpy as np
from .env_manager import BaseEnvManager
from nervex.utils import build_logger, EasyTimer


class BaseSerialActor(object):

    def __init__(self, cfg: dict) -> None:
        self._default_n_episode = cfg.get('n_episode', None)
        self._default_n_sample = cfg.get('n_sample', None)
        self._traj_len = cfg.traj_len
        if self._traj_len == "inf":
            raise ValueError(
                "Serial Actor must indicate finite traj_len, if you want to use the total episode,\
                please set this argument as the maximum length of the env episode"
            )
        self._traj_cache_length = self._traj_len
        self._traj_print_freq = cfg.traj_print_freq
        self._collect_print_freq = cfg.collect_print_freq
        self._logger, _ = build_logger(path='./log', name='actor')
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
        self._policy_output_pool = CachePool('policy_output', self._env_num)
        self._traj_cache = {env_id: deque(maxlen=self._traj_cache_length) for env_id in range(self._env_num)}
        self._total_collect_step = 0
        self._total_step = 0
        self._total_episode = 0
        self._total_sample = 0
        self._total_duration = 0

    def generate_data(self, n_episode: Optional[int] = None, n_sample: Optional[int] = None) -> Tuple[List[Any], dict]:
        assert n_episode is None or n_sample is None, "n_episode and n_sample can't be not None at the same time"
        if n_episode is not None:
            return self._collect_episode(n_episode)
        elif n_sample is not None:
            return self._collect_sample(n_sample)
        elif self._default_n_episode is not None:
            return self._collect_episode(self._default_n_episode)
        elif self._default_n_sample is not None:
            return self._collect_sample(self._default_n_sample)
        else:
            raise RuntimeError("please indicate specific n_episode or n_sample(int value)")

    def close(self) -> None:
        self._env.close()

    def _collect_episode(self, n_episode: int) -> Tuple[List[Any], dict]:
        return self._collect(lambda x, y: x >= n_episode)

    def _collect_sample(self, n_sample: int) -> Tuple[List[Any], dict]:
        return self._collect(lambda x, y: y >= n_sample)

    def _collect(self, collect_end_fn: Callable) -> Tuple[List[Any], dict]:
        episode_count = 0
        step_count = 0
        train_sample_count = 0
        episode_reward = []
        return_data = []
        info = {}
        self._policy.reset()
        with self._timer:
            while not collect_end_fn(episode_count, train_sample_count):
                obs = self._env.next_obs
                self._obs_pool.update(obs)
                env_id, obs = self._policy.data_preprocess(obs)
                policy_output = self._policy.forward(env_id, obs)
                policy_output = self._policy.data_postprocess(env_id, policy_output)
                self._policy_output_pool.update(policy_output)
                actions = {env_id: output['action'] for env_id, output in policy_output.items()}
                timesteps = self._env.step(actions)
                for env_id, timestep in timesteps.items():
                    transition = self._policy.process_transition(
                        self._obs_pool[env_id], self._policy_output_pool[env_id], timestep
                    )
                    self._traj_cache[env_id].append(transition)
                    if timestep.done or len(self._traj_cache[env_id]) == self._traj_len:
                        train_sample = self._policy.get_train_sample(self._traj_cache[env_id])
                        return_data.extend(train_sample)
                        train_sample_count += len(train_sample)
                        self._total_sample += len(train_sample)
                        if (train_sample_count + 1) % self._traj_print_freq == 0:
                            self._logger.info(
                                "env {} get new traj, collected traj: {}".format(env_id, train_sample_count)
                            )
                    if timestep.done:
                        # env reset is done by env_manager automatically
                        self._traj_cache[env_id].clear()
                        self._obs_pool.reset(env_id)
                        self._policy_output_pool.reset(env_id)
                        self._policy.reset([env_id])
                        reward = timestep.info['final_eval_reward']
                        episode_reward.append(reward)
                        self._logger.info(
                            "env {} finish episode, final reward: {}, collected episode: {}".format(
                                env_id, reward, episode_count
                            )
                        )
                        episode_count += 1
                        self._total_episode += 1
                    step_count += 1
                    self._total_step += 1
        duration = self._timer.value
        if (self._total_collect_step + 1) % self._collect_print_freq == 0:
            info = {
                'episode_count': episode_count,
                'step_count': step_count,
                'train_sample_count': train_sample_count,
                'avg_step_per_episode': step_count / max(1, episode_count),
                'avg_traj_per_epsiode': train_sample_count / max(1, episode_count),
                'avg_time_per_step': duration / (step_count + 1e-8),
                'avg_time_per_train_sample': duration / (train_sample_count + 1e-8),
                'avg_time_per_episode': duration / max(1, episode_count),
                'reward_mean': np.mean(episode_reward) if len(episode_reward) > 0 else 0.,
                'reward_std': np.std(episode_reward) if len(episode_reward) > 0 else 0.,
            }
            self._logger.info("collect end:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()])))
        self._total_collect_step += 1
        self._total_duration += duration
        collect_info = {
            'total_collect_step': self._total_collect_step,
            'total_step': self._total_step,
            'total_sample': self._total_sample,
            'total_episode': self._total_episode,
            'total_duration': self._total_duration,
        }
        return return_data, collect_info


class CachePool(object):

    def __init__(self, name: str, env_num: int, deepcopy: bool = False):
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

    def reset(self, idx: int) -> None:
        self._pool[idx] = None
