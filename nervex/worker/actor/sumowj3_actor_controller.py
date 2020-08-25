from collections import namedtuple
import time
import copy
import queue
import torch
from threading import Thread
from typing import List, Dict
from nervex.worker.actor.base_actor_controller import BaseActor
from nervex.model.sumo_dqn.sumo_dqn_network import FCDQN
from nervex.worker.agent.sumo_dqn_agent import SumoDqnActorAgent
from nervex.worker.actor.env_manager.sumowj3_env_manager import SumoWJ3EnvManager
from nervex.utils import get_step_data_compressor


class SumoWJ3Actor(BaseActor):
    # override
    def _init_with_job(self, job: dict) -> None:
        self._job = job
        self._setup_agents()
        self._setup_env_manager()
        self._compressor = get_step_data_compressor(self._job['compressor'])
        self._episode_result = []

        self._update_agent_thread = Thread(target=self._update_agent, args=())
        self._update_agent_thread.deamon = True
        self._update_agent_thread.start()  # keep alive in the whole job

    def _setup_env_manager(self) -> None:
        env_cfg = self._cfg.env
        env_cfg.env_num = self._job['env_num']
        self._env_manager = SumoWJ3EnvManager(env_cfg)

    def _setup_agents(self):
        self._agents = {}
        for name, agent_cfg in self._job['agent'].items():
            model = FCDQN(380, [2, 2, 3])
            agent = SumoDqnActorAgent(model)
            self._agents[name] = agent

    # override
    def episode_reset(self) -> None:
        for name, agent in self._agents.items():
            agent.reset()
            # load model
            agent_update_info = self.get_agent_update_info(self._job['agent'][name]['agent_update_path'])
            agent.load_state_dict(agent_update_info)
        obs = self._env_manager.reset()
        self._alive_env = [i for i in range(self._job['env_num'])]
        self._data_buffer = {k: [] for k in range(self._job['env_num'])}
        self._last_data_buffer = {k: [] for k in range(self._job['env_num'])}
        self._traj_queue = queue.Queue()
        self._episode_reward = [[] for _ in range(self._job['env_num'])]
        self._pack_trajectory_thread = Thread(target=self._pack_trajectory, args=())
        self._pack_trajectory_thread.deamon = True
        self._pack_trajectory_thread.start()
        return obs

    # override
    def __repr__(self) -> str:
        return 'SumoWJ3Actor({})'.format(self._actor_uid)

    # override
    def _agent_inference(self, obs: List[torch.Tensor]) -> dict:
        assert self._job['agent_num'] in [1]
        assert len(obs) == len(self._alive_env), len(obs)
        action, q_value = self._agents['0'].forward(obs, eps=self._job['eps'])
        data = {}
        data['action'] = action
        data['q_value'] = q_value
        return data

    # override
    def _env_step(self, agent_output: dict) -> namedtuple:
        return self._env_manager.step(agent_output['action'])

    # override
    def _accumulate_timestep(self, obs: List[Dict], agent_output: dict, timestep: namedtuple) -> None:
        for j, env_id in enumerate(self._alive_env):
            data = {
                'obs': obs[j],
                'q_value': agent_output['q_value'][j],
                'action': agent_output['action'][j],
                'reward': timestep.reward[j],
                'done': timestep.done[j],
                'priority': 1.0,
            }
            self._data_buffer[env_id].append(data)
            self._episode_reward[env_id].append(timestep.reward[j])
            if len(self._data_buffer[env_id]) == self._job['data_push_length']:
                # last data copy must be in front of obs_next
                self._last_data_buffer[env_id] = copy.deepcopy(self._data_buffer[env_id])
                handle = self._data_buffer[env_id][-1]
                self._traj_queue.put({'data': self._data_buffer[env_id], 'env_id': env_id})
                self._data_buffer[env_id] = []
            if timestep.done[j]:
                cur_len = len(self._data_buffer[env_id])
                miss_len = self._job['data_push_length'] - cur_len
                if miss_len > 0:
                    self._data_buffer[env_id] = self._last_data_buffer[env_id][-miss_len:] + self._data_buffer[env_id]
                self._traj_queue.put({'data': self._data_buffer[env_id], 'env_id': env_id})

        # deal with alive_env
        for idx, d in enumerate(timestep.done):
            if d:
                self._alive_env[idx] = -1
        self._alive_env = [e for e in self._alive_env if e != -1]

    # override
    def _finish_episode(self, timestep: namedtuple) -> None:
        assert self.all_done, 'all envs must be done'
        result = [sum(t) / (len(t) + 1e-8) for t in self._episode_reward]
        self._episode_result.append(result)
        self._logger.info(
            'finish episode{} in {} with cum_reward: {}'.format(len(self._episode_result) - 1, time.time(), result)
        )

    # override
    def _finish_job(self) -> None:
        assert len(self._episode_result) == self._job['episode_num']
        job_finish_info = {
            'job_id': self._job['job_id'],
            'actor_uid': self._actor_uid,
            'episode_num': self._job['episode_num'],
            'env_num': self._job['env_num'],
            'result': self._episode_result,
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
                for name, agent in self._agents.items():
                    agent_update_info = self.get_agent_update_info(self._job['agent'][name]['agent_update_path'])
                    agent.load_state_dict(agent_update_info)
                self._logger.info('update agent in {}'.format(time.time()))
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
                'learner_uid': self._job['learner_uid'],
                'env_id': env_id,
                'actor_uid': self._actor_uid,
                'done': data[-1]['done'],
                # TODO(nyz) the relationship between traj priority and step priority
                'priority': 1.0,
                'traj_finish_time': time.time(),
                'job_id': job_id,
                'data_push_length': self._job['data_push_length'],
                'compressor': self._job['compressor'],
                'job': self._job,
            }
            self.send_traj_metadata(metadata)
            # save data
            data = self._compressor(data)
            self.send_traj_stepdata(traj_id, data)
            self._logger.info('send traj({}) in {}'.format(traj_id, time.time()))

    # override
    @property
    def all_done(self) -> bool:
        return self._env_manager.all_done
