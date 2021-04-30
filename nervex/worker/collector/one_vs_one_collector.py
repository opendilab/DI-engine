import copy
import time
import uuid
from collections import namedtuple, deque
from threading import Thread
from functools import partial
from typing import Dict, Callable, Any, List
import numpy as np
import torch
from easydict import EasyDict

from nervex.envs import get_vec_env_setting
from nervex.torch_utils import to_device, tensor_to_list
from nervex.utils import get_data_compressor, lists_to_dicts, pretty_print, COLLECTOR_REGISTRY
from nervex.envs import BaseEnvTimestep, SyncSubprocessEnvManager, BaseEnvManager
from .base_parallel_collector import BaseCollector
from .base_serial_collector import CachePool


@COLLECTOR_REGISTRY.register('one_vs_one')
class OneVsOneCollector(BaseCollector):
    """
    Feature:
      - one policy, many envs
      - async envs(step + reset)
      - batch network eval
      - different episode length env
      - periodic policy update
      - metadata + stepdata

      - two policies
    """

    # override
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self._update_policy_thread = Thread(target=self._update_policy_periodically, args=(), name='update_policy')
        self._update_policy_thread.deamon = True

        self._start_time = time.time()
        self._traj_len = self._cfg.traj_len
        self._traj_cache_length = self._traj_len
        self._env_kwargs = self._cfg.env_kwargs
        self._compressor = get_data_compressor(self._cfg.compressor)
        self._env_manager = self._setup_env_manager()
        self._env_num = self._env_manager.env_num

        self._episode_result = [[] for k in range(self._env_num)]
        self._obs_pool = CachePool('obs', self._env_num)
        self._policy_output_pool = CachePool('policy_output', self._env_num)
        self._traj_cache = {env_id: deque(maxlen=self._traj_cache_length) for env_id in range(self._env_num)}
        self._total_step = 0
        self._total_sample = 0
        self._total_episode = 0

        self._first_update_policy = True

    def _setup_env_manager(self) -> BaseEnvManager:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(self._env_kwargs)
        manager_cfg = self._env_kwargs.get('manager', {})
        if self._eval_flag:
            env_cfg = evaluator_env_cfg
            episode_num = self._env_kwargs.evaluator_episode_num
        else:
            env_cfg = collector_env_cfg
            episode_num = self._env_kwargs.collector_episode_num
        self._episode_num = episode_num
        env_manager = SyncSubprocessEnvManager(
            env_fn=[partial(env_fn, cfg=c) for c in env_cfg], episode_num=episode_num, **manager_cfg
        )
        env_manager.launch()
        self._predefined_episode_count = episode_num * len(env_cfg)
        return env_manager

    def _start_thread(self) -> None:
        self._update_policy_thread.start()

    # override
    def close(self) -> None:
        if self._end_flag:
            return
        self._end_flag = True
        if hasattr(self, '_env_manager'):
            self._env_manager.close()

    # override
    def _policy_inference(self, obs: Dict[int, Any]) -> Dict[int, Any]:
        data_id = list(obs.keys())
        if len(self._policy) > 1:
            obs = [{id: obs[id][i] for id in data_id} for i in range(len(self._policy))]
        else:
            obs = [obs]
        assert obs[0][data_id[0]].shape == (3, 210, 160)
        self._obs_pool.update(obs)
        policy_outputs = []
        for i in range(len(self._policy)):
            if self._eval_flag:
                policy_output = self._policy[i].forward(obs)
            else:
                policy_output = self._policy[i].forward(obs, **self._cfg.collect_setting)
            policy_outputs.append(policy_output)
        self._policy_output_pool.update(policy_outputs)
        actions = {}
        for env_id in data_id:
            action = [policy_outputs[i][env_id]['action'] for i in range(len(self._policy))]
            action = torch.stack(action).squeeze()
            actions[env_id] = action
        return actions

    # override
    def _env_step(self, actions: Dict[int, Any]) -> Dict[int, Any]:
        return self._env_manager.step(actions)

    # override
    def _process_timestep(self, timestep: Dict[int, namedtuple]) -> None:
        dones = []
        for env_id, t in timestep.items():
            if t.info.get('abnormal', False):
                # If there is an abnormal timestep, reset all the related variables, also this env has been reset
                for c in self._traj_cache[env_id]:
                    c.clear()
                self._obs_pool.reset(env_id)
                self._policy_output_pool.reset(env_id)
                for p in self._policy:
                    p.reset([env_id])
                continue
            t = [BaseEnvTimestep(t.obs[i], t.reward[i], t.done, t.info) for i in range(len(self._policy))]
            if not self._eval_flag:
                for i in range(len(self._policy)):
                    if self._policy_is_active[i]:
                        # Only active policy will store transition into replay buffer.
                        transition = self._policy[i].process_transition(
                            self._obs_pool[env_id][i], self._policy_output_pool[env_id][i], t[i]
                        )
                        self._traj_cache[env_id][i].append(transition)
                full_indices = []
                for i in range(len(self._traj_cache[env_id])):
                    if len(self._traj_cache[env_id][i]) == self._traj_len:
                        full_indices.append(i)
                if t[0].done or len(full_indices) > 0:
                    for i in full_indices:
                        train_sample = self._policy[i].get_train_sample(self._traj_cache[env_id][i])
                        for s in train_sample:
                            s = self._compressor(s)
                            metadata = self._get_metadata(s, env_id)
                            self.send_stepdata(metadata['data_id'], s)
                            self.send_metadata(metadata)
                        self._total_sample += len(train_sample)
                    for c in self._traj_cache[env_id]:
                        c.clear()
            if t[0].done:
                # env reset is done by env_manager automatically
                self._obs_pool.reset(env_id)
                self._policy_output_pool.reset(env_id)
                for p in self._policy:
                    p.reset([env_id])
                reward = t[0].info['final_eval_reward']
                # Only left player's reward will be recorded.
                left_reward = reward[0]
                if isinstance(left_reward, torch.Tensor):
                    left_reward = left_reward.item()
                self._episode_result[env_id].append(left_reward)
                self.debug(
                    "Env {} finish episode, final reward: {}, collected episode: {}.".format(
                        env_id, reward, len(self._episode_result[env_id])
                    )
                )
                self._total_episode += 1
            self._total_step += 1
            dones.append(t[0].done)
        if any(dones):
            collector_info = self._get_collector_info()
            self.send_metadata(collector_info)

    # override
    def get_finish_info(self) -> dict:
        episode_count = self._episode_num * self._env_num
        duration = max(time.time() - self._start_time, 1e-8)

        game_result = copy.deepcopy(self._episode_result)
        for i, env_result in enumerate(game_result):
            for j, rew in enumerate(env_result):
                if rew < 0:
                    game_result[i][j] = "losses"
                elif rew == 0:
                    game_result[i][j] = "draws"
                else:
                    game_result[i][j] = "wins"

        finish_info = {
            'finished_task': True,  # flag
            'eval_flag': self._eval_flag,
            'episode_num': self._episode_num,
            'env_num': self._env_num,
            'duration': duration,
            'collector_done': self._env_manager.done,
            'target_episode_count': episode_count,
            'real_episode_count': self._total_episode,
            'step_count': self._total_step,
            'sample_count': self._total_sample,
            'avg_time_per_episode': duration / self._total_episode,
            'avg_time_per_step': duration / self._total_step,
            'avg_time_per_train_sample': duration / max(1, self._total_sample),
            'avg_step_per_episode': self._total_step / self._total_episode,
            'avg_sample_per_episode': self._total_sample / self._total_episode,
            'reward_mean': np.mean(self._episode_result),
            'reward_std': np.std(self._episode_result),
            'reward_raw': self._episode_result,
            'finish_time': time.time(),
            'game_result': game_result,
        }
        if not self._eval_flag:
            finish_info['collect_setting'] = self._cfg.collect_setting
        self._logger.info('\nFINISH INFO\n{}'.format(pretty_print(finish_info, direct_print=False)))
        return finish_info

    # override
    def _update_policy(self) -> None:
        path = self._cfg.policy_update_path
        self._policy_is_active = self._cfg.policy_update_flag
        for i in range(len(path)):
            if not self._first_update_policy and not self._policy_is_active[i]:
                # For the first time, all policies should be updated(i.e. initialized);
                # For other times, only active player's policies should be updated.
                continue
            while True:
                try:
                    policy_update_info = self.get_policy_update_info(path[i])
                    break
                except Exception as e:
                    self.error('Policy {} update error: {}'.format(i + 1, e))
                    time.sleep(1)

            self._policy_iter[i] = policy_update_info.pop('iter')
            self._policy[i].load_state_dict(policy_update_info)
            self.debug('Update policy {} with {}(iter{}) in {}'.format(i + 1, path, self._policy_iter, time.time()))
        self._first_update_policy = False

    # ******************************** thread **************************************

    def _update_policy_periodically(self) -> None:
        last = time.time()
        while not self._end_flag:
            cur = time.time()
            interval = cur - last
            if interval < self._cfg.policy_update_freq:
                time.sleep(self._cfg.policy_update_freq * 0.1)
                continue
            else:
                self._update_policy()
                last = time.time()
            time.sleep(0.1)

    def _get_metadata(self, stepdata: List, env_id: int) -> dict:
        data_id = "env_{}_{}".format(env_id, str(uuid.uuid1()))
        metadata = {
            'data_id': data_id,
            'env_id': env_id,
            'policy_iter': self._policy_iter,
            'unroll_len': len(stepdata),
            'compressor': self._cfg.compressor,
            'get_data_time': time.time(),
            # TODO(nyz) the relationship between traj priority and step priority
            'priority': 1.0,
        }
        return metadata

    def _get_collector_info(self) -> dict:
        return {
            'eval_flag': self._eval_flag,
            'get_info_time': time.time(),
            'collector_done': self._env_manager.done,
            'cur_episode': self._total_episode,
            'cur_sample': self._total_sample,
            'cur_step': self._total_step,
        }

    def __repr__(self) -> str:
        return "OneVsOneCollector"

    @property
    def policy(self) -> List['Policy']:  # noqa
        return self._policy

    @policy.setter
    def policy(self, _policy: List['Policy']) -> None:  # noqa
        self._policy = _policy
        if not self._eval_flag:
            for i in range(len(self._policy)):
                self._policy[i].set_setting('collect', self._cfg.collect_setting)
        self._policy_is_active = [None for _ in range(len(_policy))]
        self._policy_iter = [None for _ in range(len(_policy))]
        for env_id in self._traj_cache:
            self._traj_cache[env_id] = [deque(maxlen=self._traj_cache_length) for _ in range(len(_policy))]
