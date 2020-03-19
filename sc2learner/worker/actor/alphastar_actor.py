import time

import pysc2.env.sc2_env as sc2_env
from sc2learner.envs.alphastar_env import AlphastarEnv
from sc2learner.envs.map_info import MAPS
from sc2learner.utils import get_actor_id
from sc2learner.worker.agent import AlphastarAgent


class AlphaStarActor:
    """
    AlphaStar Actor
    Contents of each entry in  the trajectory dict
        - 'step'
        - 'agent_no'
        - 'prev_obs'
        - 'lstm_state'
        - 'actions'
        - 'next_obs'
        - 'done'
        - 'rewards'
        - 'info'
    """
    def __init__(self, cfg):
        self.cfg = cfg
        for k, v in self.cfg.rl_train.items():
            self.cfg.env[k] = v
        self.actor_id = get_actor_id()
        # env and agents are to be created after receiving job description from coordinator
        self.env = None
        self.agents = None
        self.agent_num = 0
        self._module_init()

    def _module_init(self):
        self.job_getter = JobGetter(self.cfg)
        self.model_requester = ModelRequester(self.cfg)
        self.data_pusher = DataPusher(self.cfg)

    def run_episode(self):
        job = self.job_getter.get_job(self.actor_id)
        self._init_with_job(job)
        for i in range(self.agent_num):
            model = self.model_requester.request_model(job, i)
            if isinstance(model, dict):
                self.agents[i].model.load_state_dict(model)
        obs = self.env.reset()
        data_buffer = [[]] * self.agent_num
        game_step = 0
        due = [True] * self.agent_num
        last_state_action = [None] * self.agent_num
        # main loop
        while True:
            actions = [None] * self.agent_num
            for i in range(self.agent_num):
                if due[i]:
                    act = self.agents[i].act(obs[i])
                    actions[i] = self.action_modifier(act, i)
                    last_state_action[i] = {
                        'agent_no': i,
                        'prev_obs': obs,
                        'lstm_state': self.agents[i].next_state,
                        'actions': actions
                    }
            step, due, obs, rewards, done, info = self.env.step(actions)
            game_step += step
            for i in range(self.agent_num):
                if due[i]:
                    traj_data = last_state_action[i]
                    traj_data['step'] = game_step
                    traj_data['next_obs'] = obs
                    traj_data['done'] = done
                    traj_data['rewards'] = rewards
                    traj_data['info'] = info
                    data_buffer[i].append(traj_data)
                if len(data_buffer[i]) >= job['data_push_length'] \
                        or done or game_step >= self.cfg.env.game_steps_per_episode:
                    self.data_pusher.push(job, i, data_buffer[i])
                    data_buffer[i] = []
            if done or game_step >= self.cfg.env.game_steps_per_episode:
                break

    def run(self):
        while True:
            self.run_episode()

    def action_modifier(self, act, agent_no):
        return act

    def _init_with_job(self, job):
        # TODO: set cfg.env.map_size etc. by instruction from job
        self.cfg.env.map_name = job['map_name']
        self.cfg.env.map_size = MAPS[job['map_name']][2]
        self.cfg.env.random_seed = job['random_seed']
        if job['game_type'] == 'game_vs_bot':
            self.agent_num = 1
            players = [
                sc2_env.Agent(sc2_env.Race[job['home_race']]),
                sc2_env.Bot(sc2_env.Race[job['away_race']], job['difficulty'], job['build']),
            ]
            self.agents = [AlphastarAgent(self.cfg, need_checkpoint=False)]
        elif job['game_type'] in ['self_play', 'league']:
            self.agent_num = 2
            players = [
                sc2_env.Agent(sc2_env.Race[job['home_race']]),
                sc2_env.Agent(sc2_env.Race[job['away_race']]),
            ]
            self.agents = [
                AlphastarAgent(self.cfg, need_checkpoint=False),
                AlphastarAgent(self.cfg, need_checkpoint=False)
            ]
        self.env = AlphastarEnv(self.cfg, players)


# TODO: implementation
class JobGetter:
    def __init__(self, cfg):
        self.connection = None  # TODO
        self.job_request_id = 0
        pass

    def get_job(self, actor_id):
        """
        Overview: asking for a job from the coordinator
        Input:
            - actor_id
        Output:
            - job: a dict with description of how the game should be
        """
        while True:
            job_request = {'type': 'job req', 'req_id': self.job_request_id, 'actor_id': actor_id}
            try:
                self.connection.send_pyobj(job_request)
                reply = self.job_requestor.recv_pyobj()
                assert (isinstance(reply, dict))
                if (reply['type'] != 'job'):
                    print('WARNING: received unknown response for job req, type:{}'.format(reply['type']))
                    continue
                if (reply['actor_id'] != actor_id):
                    print('WARNING: received job is assigned to another actor')
                self.job_request_id += 1
                return reply['job']
            except Exception:  # zmq.error.Again:
                print('WARNING: Job Request Timeout')
                time.sleep(1)  # wait for a while
                continue
        self.job_request_id += 1


class ModelRequester:
    def __init__(self, cfg):
        pass

    def request_model(self, job, agent_no):
        """
        Overview: fetch a model from somewhere
        Input:
            - job: a dict with description of how the game should be
            - agent_no: 0 or 1, labeling the two agents of game
        """
        pass


class DataPusher:
    def __init__(self, cfg):
        pass

    def push(self, job, agent_no, data_buffer):
        pass
