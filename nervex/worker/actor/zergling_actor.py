import copy
import queue
import time
from collections import namedtuple
from threading import Thread
from typing import List, Dict, Callable, Any

import torch

from nervex.data import default_collate, default_decollate
from nervex.torch_utils import to_device, tensor_to_list
from nervex.utils import get_step_data_compressor
from nervex.worker.actor import BaseActor
from nervex.worker.actor.env_manager import SubprocessEnvManager


class ZerglingActor(BaseActor):
    """
    Feature:
      - one agent, many envs
      - async envs(step + reset)
      - batch network eval
      - different episode length env
      - periodic agent update
      - metadata + stepdata
    """

    # override
    def _init_with_job(self, job: dict) -> None:
        super()._init_with_job(job)
        self._job = job
        assert len(self._job['agent']) == 1
        self._setup_env_manager()
        self._setup_agent()
        self._compressor = get_step_data_compressor(self._job['compressor'])
        self._job_result = {k: [] for k in range(self._job['env_num'])}
        self._collate_fn = default_collate
        self._decollate_fn = default_decollate
        # init agent(reset agent, load model)
        self._agent.reset()
        path = self._job['agent'][self._agent_name]['agent_update_path']
        agent_update_info = self.get_agent_update_info(path)
        self._agent.load_state_dict(agent_update_info)
        # init env
        self._env_manager.launch()
        self._obs_pool = {k: None for k in range(self._job['env_num'])}
        self._act_pool = {k: None for k in range(self._job['env_num'])}
        self._data_buffer = {k: [] for k in range(self._job['env_num'])}
        self._last_data_buffer = {k: [] for k in range(self._job['env_num'])}
        self._traj_queue = queue.Queue()
        self._episode_result = {k: None for k in range(self._job['env_num'])}

        self._update_agent_thread = Thread(target=self._update_agent, args=())
        self._update_agent_thread.deamon = True
        self._update_agent_thread.start()  # keep alive in the whole job
        self._pack_trajectory_thread = Thread(target=self._pack_trajectory, args=())
        self._pack_trajectory_thread.deamon = True
        self._pack_trajectory_thread.start()

    def _setup_env_manager(self) -> None:
        env_cfg = self._cfg.env
        self._setup_env_fn(env_cfg)
        env_num = self._job['env_num']
        self._env_manager = SubprocessEnvManager(
            env_fn=self._env_fn,
            env_cfg=[env_cfg for _ in range(env_num)],
            env_num=env_num,
            episode_num=self._job['episode_num']
        )

    # override
    def _agent_inference(self, obs: Dict[int, Any]) -> Dict[int, Any]:
        # save in obs_pool
        for k, v in obs.items():
            self._obs_pool[k] = copy.deepcopy(v)

        env_id = obs.keys()
        obs = self._collate_fn(list(obs.values()))
        if self._cfg.actor.use_cuda:
            obs = to_device(obs, 'cuda')
        data = self._agent.forward(obs, **self._job['forward_kwargs'])
        if self._cfg.actor.use_cuda:
            data = to_device(data, 'cpu')
        data = self._decollate_fn(data)
        data = {i: d for i, d in zip(env_id, data)}
        return data

    # override
    def _env_step(self, agent_output: Dict[int, Dict]) -> Dict[int, Any]:
        # save in act_pool
        for k, v in agent_output.items():
            self._act_pool[k] = copy.deepcopy(v)
        action = {k: v['action'] for k, v in agent_output.items()}
        return self._env_manager.step(action)

    # override
    def _process_timestep(self, timestep: Dict[int, namedtuple]) -> None:
        for env_id, t in timestep.items():
            data = self._get_transition(self._obs_pool[env_id], self._act_pool[env_id], timestep[env_id])
            self._data_buffer[env_id].append(data)
            if len(self._data_buffer[env_id]) == self._job['data_push_length']:
                # last data copy must be in front of obs_next
                self._last_data_buffer[env_id] = copy.deepcopy(self._data_buffer[env_id])
                handle = self._data_buffer[env_id][-1]
                self._traj_queue.put({'data': self._data_buffer[env_id], 'env_id': env_id})
                self._data_buffer[env_id] = []
            if t.done:
                self._job_result[env_id].append(t.info)
                self._logger.info('env{} finish episode in {}'.format(env_id, time.time()))
                cur_len = len(self._data_buffer[env_id])
                miss_len = self._job['data_push_length'] - cur_len
                if miss_len > 0:
                    self._data_buffer[env_id] = self._last_data_buffer[env_id][-miss_len:] + self._data_buffer[env_id]
                self._traj_queue.put({'data': self._data_buffer[env_id], 'env_id': env_id})

    # override
    def _finish_job(self) -> None:
        assert all([len(r) == self._job['episode_num'] for r in self._job_result.values()])
        job_finish_info = {
            'job_id': self._job['job_id'],
            'actor_uid': self._actor_uid,
            'episode_num': self._job['episode_num'],
            'env_num': self._job['env_num'],
            'player_id': self._job['player_id'],
            'launch_player': self._job['launch_player'],
            'result': tensor_to_list(self._job_result),
        }
        self.send_finish_job(job_finish_info)

    # ******************************** thread **************************************

    # override
    def _update_agent(self) -> None:
        last = time.time()
        while not self._end_flag:
            cur = time.time()
            interval = cur - last
            if interval < self._job['agent_update_freq']:
                time.sleep(self._job['agent_update_freq'] * 0.1)
                continue
            else:
                path = self._job['agent'][self._agent_name]['agent_update_path']
                agent_update_info = self.get_agent_update_info(path)
                self._agent.load_state_dict(agent_update_info)
                self._logger.info('update agent in {} with path {}'.format(time.time(), path))
                last = time.time()

    # override
    def _pack_trajectory(self) -> None:
        while not self._end_flag:
            try:
                data = self._traj_queue.get()
            except queue.Empty:
                time.sleep(3)
                continue
            data, env_id = list(data.values())
            # send metadata
            job_id = self._job['job_id']
            traj_id = "job_{}_env_{}".format(job_id, env_id)
            metadata = {
                'traj_id': traj_id,
                'learner_uid': self._job['learner_uid'][0],
                'launch_player': self._job['launch_player'],
                'env_id': env_id,
                'actor_uid': self._actor_uid,
                'done': data[-1]['done'],
                # TODO(nyz) the relationship between traj priority and step priority
                'priority': 1.0,
                'traj_finish_time': time.time(),
                'job_id': job_id,
                'data_push_length': len(data),
                'compressor': self._job['compressor'],
                'job': self._job,
            }
            # save data
            data = self._compressor(data)
            self.send_traj_stepdata(traj_id, data)
            self.send_traj_metadata(metadata)
            self._logger.info('send traj({}) in {}'.format(traj_id, time.time()))

    def _setup_env_fn(self, env_cfg: dict) -> None:
        """set self._env_fn"""
        raise NotImplementedError

    def _setup_agent(self) -> None:
        """set self._agent, self._agent_name"""
        raise NotImplementedError

    def _get_transition(self, obs: Any, agent_output: Dict, timestep: namedtuple) -> dict:
        """get one step transition"""
        raise NotImplementedError
