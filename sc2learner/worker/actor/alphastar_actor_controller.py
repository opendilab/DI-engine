from collections import namedtuple
from sc2learner.worker.actor import BaseActor
from sc2learner.data import BaseContainer
from sc2learner.agent import AlphaStarAgent
from sc2learner.utils import EasyTimer, get_actor_uid


class AlphaStarActor(BaseActor):
    # override
    def __init__(self, cfg: dict):
        super(AlphaStarActor, self).__init__(cfg)
        self._actor_uid = get_actor_uid()
        self._timer = EasyTimer()

    # override
    def _init_with_job(self, job: dict) -> None:
        self._job = job
        self._agent = None
        self._teacher_agent = None
        self._env_manager = None

    # override
    def episode_reset(self) -> None:
        for a in self._agents:
            a.reset()
        obs = self._env_manager.reset()
        self._alive_env = [i for i in range(self._job['env_num'])]
        self._data_buffer = {k: {k1: [] for k1 in self._job['env_num']} for k in self._job['agent_num']}
        return obs

    # override
    def __repr__(self) -> str:
        return 'AlphaStarActor({})'.format(self.actor_uid)

    # override
    def _agent_inference(self, obs: list) -> dict:
        assert self._job['agent_num'] in [2, 3, 4]  # 1v1, whether with teacher agent
        action = [[] for _ in range(self._job['agent_num'])]
        hidden_state = [[] for _ in range(self._job['agent_num'])]
        for agent_idx, agent in self._agents:
            act, h = agent(obs[agent_idx % 2])
            action.append(act)
            hidden_state.append(h)
        data = {}
        data['action'] = action[:2]
        data['prev_state'] = hidden_state[:2]
        if self._job['agent_num'] > 2:
            data['teacher_action'] = action[2:]
            data['teacher_prev_state'] = hidden_state[2:]
        return action['action']

    # override
    def _env_step(self, action: dict) -> namedtuple:
        return self._env_manager.step(action['action'])

    # override
    def _accumulate_timestep(self, obs: list, action: dict, timestep: namedtuple) -> None:
        for i in range(len(self._job['env_agent_num'])):
            for j, env_id in enumerate(self._alive_env):
                action = action['action'][i][j]
                # only valid action step can be pushed into data_buffer
                if action is not None:
                    data = {'obs_home': obs[i][j]}
                    if len(self._job['agent_num']) == 2:
                        data.update({'obs_away': obs[1 - i][j]})
                    data.update(
                        {
                            'action': action['action'][i][j],
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
                                'teacher_prev_state': action['teacher_prev_state'][i][j],
                            }
                        )
                    self._data_buffer[i][env_id].append(data)
                # due is related to next_obs
                due = timestep.due[i][j]
                done = timestep.done[i][j]
                if done or (due and len(self._data_buffer[i][env_id]) == self._job['data_push_length']):
                    handle = self._data_buffer[i][env_id][-1]
                    handle['obs_home_next'] = timestep.obs[i][j]
                    handle['obs_away_next'] = timestep.obs[i][1 - j]
        # deal with alive_env
        for idx, d in enumerate(timestep.done):
            if d:
                self._alive_env[idx] = -1
        self._alive_env = [e for e in self._alive_env if e != -1]
