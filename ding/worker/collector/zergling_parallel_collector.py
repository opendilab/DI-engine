from typing import Dict, Any, List
import time
import uuid
from collections import namedtuple
from threading import Thread
from functools import partial

import numpy as np
import torch
from easydict import EasyDict

from ding.policy import create_policy, Policy
from ding.envs import get_vec_env_setting, create_env_manager, BaseEnvManager
from ding.utils import get_data_compressor, pretty_print, PARALLEL_COLLECTOR_REGISTRY
from .base_parallel_collector import BaseParallelCollector
from .base_serial_collector import CachePool, TrajBuffer

INF = float("inf")


@PARALLEL_COLLECTOR_REGISTRY.register('zergling')
class ZerglingParallelCollector(BaseParallelCollector):
    """
    Feature:
      - one policy, many envs
      - async envs(step + reset)
      - batch network eval
      - different episode length env
      - periodic policy update
      - metadata + stepdata
    """
    config = dict(
        print_freq=5,
        compressor='lz4',
        update_policy_second=3,
        # The following keys is set by the commander
        # env
        # policy
        # collect_setting
        # eval_flag
        # policy_update_path
    )

    # override
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self._update_policy_thread = Thread(
            target=self._update_policy_periodically, args=(), name='update_policy', daemon=True
        )
        self._start_time = time.time()
        self._compressor = get_data_compressor(self._cfg.compressor)

        # create env
        self._env_cfg = self._cfg.env
        env_manager = self._setup_env_manager(self._env_cfg)
        self.env_manager = env_manager

        # create policy
        if self._eval_flag:
            policy = create_policy(self._cfg.policy, enable_field=['eval']).eval_mode
        else:
            policy = create_policy(self._cfg.policy, enable_field=['collect']).collect_mode
        self.policy = policy

        self._episode_result = [[] for k in range(self._env_num)]
        self._obs_pool = CachePool('obs', self._env_num)
        self._policy_output_pool = CachePool('policy_output', self._env_num)
        self._traj_buffer = {env_id: TrajBuffer(self._traj_len) for env_id in range(self._env_num)}
        self._total_step = 0
        self._total_sample = 0
        self._total_episode = 0

    @property
    def policy(self) -> Policy:
        return self._policy

    # override
    @policy.setter
    def policy(self, _policy: Policy) -> None:
        self._policy = _policy
        self._n_episode = _policy.get_attribute('cfg').collect.get('n_episode', None)
        self._n_sample = _policy.get_attribute('cfg').collect.get('n_sample', None)
        assert any(
            [t is None for t in [self._n_sample, self._n_episode]]
        ), "n_episode/n_sample in policy cfg can't be not None at the same time"
        # TODO(nyz) the same definition of traj_len in serial and parallel
        if self._n_episode is not None:
            self._traj_len = INF
        elif self._n_sample is not None:
            self._traj_len = self._n_sample

    @property
    def env_manager(self, _env_manager) -> None:
        self._env_manager = _env_manager

    # override
    @env_manager.setter
    def env_manager(self, _env_manager: BaseEnvManager) -> None:
        self._env_manager = _env_manager
        self._env_manager.launch()
        self._env_num = self._env_manager.env_num
        self._predefined_episode_count = self._env_num * self._env_manager._episode_num

    def _setup_env_manager(self, cfg: EasyDict) -> BaseEnvManager:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg)
        if self._eval_flag:
            env_cfg = evaluator_env_cfg
        else:
            env_cfg = collector_env_cfg
        env_manager = create_env_manager(cfg.manager, [partial(env_fn, cfg=c) for c in env_cfg])
        return env_manager

    def _start_thread(self) -> None:
        # evaluator doesn't need to update policy periodically, only updating policy when starts
        if not self._eval_flag:
            self._update_policy_thread.start()

    def _join_thread(self) -> None:
        if not self._eval_flag:
            self._update_policy_thread.join()
            del self._update_policy_thread

    # override
    def close(self) -> None:
        if self._end_flag:
            return
        self._end_flag = True
        time.sleep(1)
        if hasattr(self, '_env_manager'):
            self._env_manager.close()
        self._join_thread()

    # override
    def _policy_inference(self, obs: Dict[int, Any]) -> Dict[int, Any]:
        self._obs_pool.update(obs)
        if self._eval_flag:
            policy_output = self._policy.forward(obs)
        else:
            policy_output = self._policy.forward(obs, **self._cfg.collect_setting)
        self._policy_output_pool.update(policy_output)
        actions = {env_id: output['action'] for env_id, output in policy_output.items()}
        return actions

    # override
    def _env_step(self, actions: Dict[int, Any]) -> Dict[int, Any]:
        return self._env_manager.step(actions)

    # override
    def _process_timestep(self, timestep: Dict[int, namedtuple]) -> None:
        send_data_time = []
        for env_id, t in timestep.items():
            if t.info.get('abnormal', False):
                # if there is a abnormal timestep, reset all the related variable, also this env has been reset
                self._traj_buffer[env_id].clear()
                self._obs_pool.reset(env_id)
                self._policy_output_pool.reset(env_id)
                self._policy.reset([env_id])
                continue
            self._total_step += 1
            if t.done:  # must be executed before send_metadata
                self._total_episode += 1
            if not self._eval_flag:
                transition = self._policy.process_transition(
                    self._obs_pool[env_id], self._policy_output_pool[env_id], t
                )
                self._traj_buffer[env_id].append(transition)
            if (not self._eval_flag) and (t.done or len(self._traj_buffer[env_id]) == self._traj_len):
                train_sample = self._policy.get_train_sample(self._traj_buffer[env_id])
                for s in train_sample:
                    s = self._compressor(s)
                    self._total_sample += 1
                    with self._timer:
                        metadata = self._get_metadata(s, env_id)
                        object_ref = self.send_stepdata(metadata['data_id'], s)
                        if object_ref:
                            metadata['object_ref'] = object_ref
                        self.send_metadata(metadata)
                    send_data_time.append(self._timer.value)
                self._traj_buffer[env_id].clear()
            if t.done:
                # env reset is done by env_manager automatically
                self._obs_pool.reset(env_id)
                self._policy_output_pool.reset(env_id)
                self._policy.reset([env_id])
                reward = t.info['final_eval_reward']
                if isinstance(reward, torch.Tensor):
                    reward = reward.item()
                self._episode_result[env_id].append(reward)
                self.debug(
                    "env {} finish episode, final reward: {}, collected episode {}".format(
                        env_id, reward, len(self._episode_result[env_id])
                    )
                )
        self.debug(
            "send {} train sample with average time: {:.6f}".format(
                len(send_data_time),
                sum(send_data_time) / (1e-6 + len(send_data_time))
            )
        )
        dones = [t.done for t in timestep.values()]
        if any(dones):
            collector_info = self._get_collector_info()
            self.send_metadata(collector_info)

    # override
    def get_finish_info(self) -> dict:
        duration = max(time.time() - self._start_time, 1e-8)
        episode_result = sum(self._episode_result, [])
        finish_info = {
            'eval_flag': self._eval_flag,
            'env_num': self._env_num,
            'duration': duration,
            'train_iter': self._policy_iter,
            'collector_done': self._env_manager.done,
            'predefined_episode_count': self._predefined_episode_count,
            'real_episode_count': self._total_episode,
            'step_count': self._total_step,
            'sample_count': self._total_sample,
            'avg_time_per_episode': duration / max(1, self._total_episode),
            'avg_time_per_step': duration / self._total_step,
            'avg_time_per_train_sample': duration / max(1, self._total_sample),
            'avg_step_per_episode': self._total_step / max(1, self._total_episode),
            'avg_sample_per_episode': self._total_sample / max(1, self._total_episode),
            'reward_mean': np.mean(episode_result) if len(episode_result) > 0 else 0,
            'reward_std': np.std(episode_result) if len(episode_result) > 0 else 0,
            'reward_raw': episode_result,
            'finish_time': time.time()
        }
        if not self._eval_flag:
            finish_info['collect_setting'] = self._cfg.collect_setting
        self._logger.info('\nFINISH INFO\n{}'.format(pretty_print(finish_info, direct_print=False)))
        return finish_info

    # override
    def _update_policy(self) -> None:
        path = self._cfg.policy_update_path
        while True:
            try:
                policy_update_info = self.get_policy_update_info(path)
                break
            except Exception as e:
                self.error('Policy update error: {}'.format(e))
                time.sleep(1)
        if policy_update_info is None:
            return

        self._policy_iter = policy_update_info.pop('iter')
        self._policy.load_state_dict(policy_update_info)
        self.debug('update policy with {}(iter{}) in {}'.format(path, self._policy_iter, time.time()))

    # ******************************** thread **************************************

    def _update_policy_periodically(self) -> None:
        last = time.time()
        while not self._end_flag:
            cur = time.time()
            interval = cur - last
            if interval < self._cfg.update_policy_second:
                time.sleep(self._cfg.update_policy_second * 0.1)
                continue
            else:
                self._update_policy()
                last = time.time()
            time.sleep(0.1)

    def _get_metadata(self, stepdata: List, env_id: int) -> dict:
        data_id = "env_{}_{}".format(env_id, str(uuid.uuid1()))
        metadata = {
            'eval_flag': self._eval_flag,
            'data_id': data_id,
            'env_id': env_id,
            'policy_iter': self._policy_iter,
            'unroll_len': len(stepdata),
            'compressor': self._cfg.compressor,
            'get_data_time': time.time(),
            # TODO(nyz) the relationship between traj priority and step priority
            'priority': 1.0,
            'cur_episode': self._total_episode,
            'cur_sample': self._total_sample,
            'cur_step': self._total_step,
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
        return "ZerglingParallelCollector"
