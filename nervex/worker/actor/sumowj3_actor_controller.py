import copy
import queue
import time
from collections import namedtuple
from threading import Thread
from typing import List, Dict

import torch

from nervex.envs.sumo import SumoWJ3Env, FakeSumoWJ3Env
from nervex.model import FCDQN
from nervex.torch_utils import tensor_to_list, to_device
from nervex.utils import get_step_data_compressor
from nervex.worker.actor import BaseActor, register_actor
from nervex.worker.actor.env_manager import SubprocessEnvManager
from nervex.worker.agent.sumo_dqn_agent import SumoDqnActorAgent


class SumoWJ3Actor(BaseActor):
    # override
    def _init_with_job(self, job: dict) -> None:
        super()._init_with_job(job)
        self._job = job
        self._setup_env_manager()
        self._setup_agents()
        self._compressor = get_step_data_compressor(self._job['compressor'])
        self._episode_result = []

        self._update_agent_thread = Thread(target=self._update_agent, args=())
        self._update_agent_thread.deamon = True
        self._update_agent_thread.start()  # keep alive in the whole job

    def _setup_env_manager(self) -> None:
        env_cfg = self._cfg.env
        env_fn_mapping = {'normal': SumoWJ3Env, 'fake': FakeSumoWJ3Env}
        env_fn = env_fn_mapping[env_cfg.env_type]
        env_num = self._job['env_num']
        self._env_manager = SubprocessEnvManager(
            env_fn=env_fn, env_cfg=[env_cfg for _ in range(env_num)], env_num=env_num
        )

    def _setup_agents(self):
        self._agents = {}
        env_info = self._env_manager._envs[0].info()
        for name, agent_cfg in self._job['agent'].items():
            model = FCDQN(env_info.obs_space.shape, list(env_info.act_space.shape.values()))
            if self._cfg.actor.use_cuda:
                model.cuda()
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
        self._one_episode_cum_reward = [None for _ in range(self._job['env_num'])]
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
        if self._cfg.actor.use_cuda:
            obs = to_device(obs, 'cuda')
        obs = torch.stack(obs, dim=0)
        action, q_value = self._agents['0'].forward(obs, eps=self._job['eps'])
        data = {'action': action, 'q_value': q_value}
        if self._cfg.actor.use_cuda:
            obs = to_device(obs, 'cpu')
        return data

    # override
    def _env_step(self, agent_output: dict) -> namedtuple:
        return self._env_manager.step(agent_output['action'])

    # override
    def _accumulate_timestep(self, obs: List[Dict], agent_output: dict, timestep: namedtuple) -> None:
        for j, env_id in enumerate(self._alive_env):
            data = {
                'obs': obs[j],
                'next_obs': timestep.obs[j],
                'q_value': agent_output['q_value'][j],
                'action': agent_output['action'][j],
                'reward': timestep.reward[j],
                'done': timestep.done[j],
                'priority': 1.0,
            }
            self._data_buffer[env_id].append(data)
            if len(self._data_buffer[env_id]) == self._job['data_push_length']:
                # last data copy must be in front of obs_next
                self._last_data_buffer[env_id] = copy.deepcopy(self._data_buffer[env_id])
                handle = self._data_buffer[env_id][-1]
                self._traj_queue.put({'data': self._data_buffer[env_id], 'env_id': env_id})
                self._data_buffer[env_id] = []
            if timestep.done[j]:
                self._one_episode_cum_reward[env_id] = tensor_to_list(timestep.info[j]['cum_reward'])
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
        self._episode_result.append(self._one_episode_cum_reward)
        self._logger.info(
            'finish episode{} in {} with cum_reward: {}'.format(
                len(self._episode_result) - 1, time.time(), self._episode_result[-1]
            )
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

    # override
    @property
    def all_done(self) -> bool:
        return self._env_manager.all_done


register_actor("sumowj3", SumoWJ3Actor)
