from collections import namedtuple
import time
import copy
import queue
from threading import Thread
from typing import List, Dict
from nervex.worker.actor.base_actor_controller import BaseActor
from nervex.model import AlphaStarActorCritic
from nervex.worker.agent.alphastar_agent import AlphaStarAgent
import nervex.worker.actor.comm.as_comm_actor
from nervex.worker.actor.env_manager.alphastar_env_manager import AlphaStarEnvManager
from nervex.utils import get_step_data_compressor


class AlphaStarActor(BaseActor):
    # override
    def _check(self) -> None:
        super()._check()
        hasattr(self, 'get_env_stat')

    # override
    def _init_with_job(self, job: dict) -> None:
        self._job = job
        self._setup_agents()
        self._setup_env_manager()
        self._compressor = get_step_data_compressor(self._job['compressor'])
        self._stat = [self.get_env_stat(p) for p in self._job['stat_path']]
        self._episode_result = []

        self._update_agent_thread = Thread(target=self._update_agent, args=())
        self._update_agent_thread.start()  # keep alive in the whole job

    def _setup_env_manager(self) -> None:
        raise NotImplementedError

    def _setup_agents(self):
        self._agents = {}
        for name, agent_cfg in self._job['agent'].items():
            model = AlphaStarActorCritic(agent_cfg['model'])
            agent = AlphaStarAgent(model, self._job['env_num'])
            self._agents[name] = agent

    # override
    def episode_reset(self) -> None:
        for name, agent in self._agents.items():
            # reset state(e.g. lstm hidden state)
            agent.reset()
            # load model
            agent_update_info = self.get_agent_update_info(self._job['agent'][name]['agent_update_path'])
            agent.load_state_dict(agent_update_info)
        obs = self._env_manager.reset(reset_param=[{'loaded_stat': self._stat} for _ in range(self._job['env_num'])])
        self._alive_env = [i for i in range(self._job['env_num'])]
        self._data_buffer = {k: {k1: [] for k1 in range(self._job['env_num'])} for k in range(self._job['send_data_agent_num'])}
        self._last_data_buffer = {k: {k1: [] for k1 in range(self._job['env_num'])} for k in range(self._job['send_data_agent_num'])}
        self._traj_queue = queue.Queue()
        return obs

    # override
    def __repr__(self) -> str:
        return 'AlphaStarActor({})'.format(self._actor_uid)

    # override
    def _agent_inference(self, obs: List[Dict]) -> dict:
        assert self._job['agent_num'] in [2, 3, 4]  # 1v1, whether with teacher agent
        assert len(obs) == 2, len(obs)
        env_action, algo_action, action_output, hidden_state = [], [], [], []
        for agent_obs_idx, agent in zip(self._job['agent_obs_idx'], self._agents.values()):
            inputs = {'data': obs[agent_obs_idx], 'state_info': {i: False for i in self._alive_env}}
            output = agent.forward(inputs, param={'mode': 'evaluate'})
            env_action.append(output['env_action'])
            algo_action.append(output['algo_action'])
            action_output.append(output['action_output'])
            hidden_state.append(output['prev_state'])
        data = {}
        data['env_action'] = [env_action[i] for i in self._job['agent_idx']]
        data['algo_action'] = [algo_action[i] for i in self._job['agent_idx']]
        data['action_output'] = [action_output[i] for i in self._job['agent_idx']]
        data['prev_state'] = [hidden_state[i] for i in self._job['agent_idx']]
        if 'teacher_agent_idx' in self._job.keys():
            data['teacher_action'] = [algo_action[i] for i in self._job['teacher_agent_idx']]
            data['teacher_action_output'] = [action_output[i] for i in self._job['teacher_agent_idx']]
            data['teacher_prev_state'] = [hidden_state[i] for i in self._job['teacher_agent_idx']]
        return data

    # override
    def _env_step(self, action: dict) -> namedtuple:
        return self._env_manager.step(action['env_action'])

    # override
    def _accumulate_timestep(self, obs: List[Dict], action: dict, timestep: namedtuple) -> None:
        for i in range(self._job['send_data_agent_num']):
            for j, env_id in enumerate(self._alive_env):
                test_action = action['algo_action'][i][j]
                # only valid action step can be pushed into data_buffer
                if test_action is not None:
                    data = {'obs_home': obs[i][j]}
                    data.update({'obs_away': obs[1 - i][j]})
                    data.update(
                        {
                            'action': action['algo_action'][i][j],
                            'action_output': action['action_output'][i][j],
                            'prev_state': action['prev_state'][i][j],
                            'reward': timestep.reward[i][j],
                            'done': timestep.done[i][j],
                            'step': timestep.episode_steps[i][j],
                            # TODO(nyz): implement other priority initialization algo, setting it to 1.0 now
                            'priority': 1.0,
                        }
                    )
                    if 'teacher_action' in action.keys():
                        data.update(
                            {
                                'teacher_action': action['teacher_action'][i][j],
                                'teacher_action_output': action['teacher_action_output'][i][j],
                                'teacher_prev_state': action['teacher_prev_state'][i][j],
                            }
                        )
                    self._data_buffer[i][env_id].append(data)
                # due is related to next_obs
                due = timestep.due[i][j]
                done = timestep.done[i][j]
                if due and len(self._data_buffer[i][env_id]) == self._job['data_push_length']:
                    # last data copy must be in front of obs_next
                    self._last_data_buffer[i][env_id] = copy.deepcopy(self._data_buffer[i][env_id])
                    handle = self._data_buffer[i][env_id][-1]
                    handle['obs_home_next'] = timestep.obs[i][j]
                    handle['obs_away_next'] = timestep.obs[1 - i][j]
                    self._traj_queue.put(self._data_buffer[i][env_id])
                    self._data_buffer[i][env_id] = []
                if done:
                    cur_len = len(self._data_buffer[i][env_id])
                    miss_len = self._job['data_push_length'] - cur_len
                    if miss_len > 0:
                        self._data_buffer[i][
                            env_id] = self._last_data_buffer[i][env_id][-miss_len:] + self._data_buffer[i][env_id]
                    handle = self._data_buffer[i][env_id][-1]
                    handle['obs_home_next'] = timestep.obs[i][j]
                    handle['obs_away_next'] = timestep.obs[1 - i][j]
                    self._traj_queue.put({'data': self._data_buffer[i][env_id], 'env_id': env_id, 'agent_id': i})

        # deal with alive_env
        for idx, d in enumerate(timestep.done):
            if any(d):
                self._alive_env[idx] = -1
        self._alive_env = [e for e in self._alive_env if e != -1]

    # override
    def _finish_episode(self, timestep: namedtuple) -> None:
        assert self.all_done, 'all envs must be done'
        result_map = {1: 'wins', 0: 'draws', -1: 'losses'}
        result = [result_map[timestep.reward[0][j]['winloss'].int().item()] for j in range(self._job['env_num'])]
        self._episode_result.append(result)
        self._logger.info('finish episode{} in {}'.format(len(self._episode_result) - 1, time.time()))

    # override
    def _finish_job(self) -> None:
        assert len(self._episode_result) == self._job['episode_num']
        job_finish_info = {
            'job_id': self._job['job_id'],
            'actor_uid': self._actor_uid,
            'player_id': self._job['player_id'],
            'episode_num': self._job['episode_num'],
            'env_num': self._job['env_num'],
            'result': self._episode_result,
        }
        self.send_finish_job(job_finish_info)

    # ******************************** thread **************************************

    # override
    def _update_agent(self) -> None:
        last = time.time()
        while True:
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
        while True:
            try:
                data = self._traj_queue.get()
            except queue.Empty:
                time.sleep(3)
                continue
            data, env_id, agent_id = list(data.values())
            # send metadata
            end_game_step = data[-1]['step']
            job_id = self._job['job_id']
            traj_id = "job_{}_agent_{}_env_{}_step_{}".format(job_id, agent_id, env_id, end_game_step)
            metadata = {
                'traj_id': traj_id,
                'player_id': self._job['player_id'][agent_id],
                'agent_id': agent_id,
                'env_id': env_id,
                'actor_uid': self._actor_uid,
                'game_step': end_game_step,
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
