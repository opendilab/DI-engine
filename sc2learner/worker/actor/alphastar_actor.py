import time
import copy
import pickle

import torch

import pysc2.env.sc2_env as sc2_env
from sc2learner.agent.alphastar_agent import AlphaStarAgent
from sc2learner.envs.alphastar_env import AlphaStarEnv
from sc2learner.utils import get_actor_uid, dict_list2list_dict, merge_two_dicts, get_step_data_compressor
from sc2learner.torch_utils import to_device


def unsqueeze_batch_dim(obs):
    # helper function for the evaluation of a single observation
    def unsqueeze(x):
        if isinstance(x, dict):
            for k in x.keys():
                if isinstance(x[k], dict):
                    for kk in x[k].keys():
                        x[k][kk] = x[k][kk].unsqueeze(0)
                else:
                    x[k] = x[k].unsqueeze(0)
        elif isinstance(x, torch.Tensor):
            x = x.unsqueeze(0)
        else:
            raise TypeError("invalid type: {}".format(type(x)))
        return x

    unsqueeze_keys = ['scalar_info', 'spatial_info']
    list_keys = ['entity_info', 'entity_raw', 'map_size']
    for k, v in obs.items():
        if k in unsqueeze_keys:
            obs[k] = unsqueeze(v)
        if k in list_keys:
            obs[k] = [obs[k]]
    return obs


class AlphaStarActor:
    def __init__(self, cfg):
        self.cfg = cfg
        # copying everything in rl_train and train config entry to config.env
        # TODO: better handling for common variables used in different situations
        if 'rl_train' in self.cfg:
            for k, v in self.cfg.rl_train.items():
                self.cfg.env[k] = v
        if 'train' in self.cfg:
            for k, v in self.cfg.train.items():
                self.cfg.env[k] = v
        # in case we want everything to be the default
        if 'model' not in self.cfg:
            self.cfg.model = None
        self.actor_uid = get_actor_uid()
        # env and agents are to be created after receiving job description from coordinator
        self.env = None
        self.agents = None
        self.agent_num = 0
        self.teacher_agent = None
        self.use_teacher_model = None
        self.compressor_name = None
        self.compressor = None

    def _init_with_job(self, job):
        # preparing the environment with the received job description
        self.cfg.env.map_name = job['map_name']
        self.cfg.env.random_seed = job['random_seed']

        if job['game_type'] == 'game_vs_bot':
            self.agent_num = 1
            players = [
                sc2_env.Agent(sc2_env.Race[job['home_race']]),
                sc2_env.Bot(
                    sc2_env.Race[job['away_race']], sc2_env.Difficulty[job['difficulty']],
                    sc2_env.BotBuild[job['build']] if 'build' in job else None
                ),
            ]
            self.agents = [
                AlphaStarAgent(
                    model_config=self.cfg.model,
                    num_concurrent_episodes=1,
                    use_cuda=self.cfg.env.use_cuda,
                    use_distributed=False
                )
            ]
        elif job['game_type'] in ['self_play', 'league']:
            self.agent_num = 2
            players = [
                sc2_env.Agent(sc2_env.Race[job['home_race']]),
                sc2_env.Agent(sc2_env.Race[job['away_race']]),
            ]
            self.agents = [
                AlphaStarAgent(
                    model_config=self.cfg.model,
                    num_concurrent_episodes=1,
                    use_cuda=self.cfg.env.use_cuda,
                    use_distributed=False
                ),
                AlphaStarAgent(
                    model_config=self.cfg.model,
                    num_concurrent_episodes=1,
                    use_cuda=self.cfg.env.use_cuda,
                    use_distributed=False
                )
            ]
        else:
            raise NotImplementedError()

        for agent in self.agents:
            agent.eval()
            agent.set_seed(job['random_seed'])
            agent.reset_previous_state([True])  # Here the lstm_states are reset

        if job['teacher_model_id']:
            # agent for evaluation of the SL model to produce the teacher_logits
            # for human_policy_kl_loss in rl.py of AlphaStar Supp. Mat.
            self.teacher_agent = AlphaStarAgent(
                model_config=self.cfg.model,
                num_concurrent_episodes=1,
                use_cuda=self.cfg.env.use_cuda,
                use_distributed=False
            )
            self.teacher_agent.eval()
            self.teacher_agent.set_seed(job['random_seed'])
            self.teacher_agent.reset_previous_state([True])
            self.use_teacher_model = True
        else:
            self.use_teacher_model = False
        self.env = self._make_env(players)
        self.compressor_name = job['step_data_compressor']
        self.compressor = get_step_data_compressor(self.compressor_name)

    def _make_env(self, players):
        return AlphaStarEnv(self.cfg, players)

    # this is to be overridden in real worker or evaluator classes
    def _module_init(self):
        self.job_getter = JobGetter(self.cfg)
        self.model_loader = ModelLoader(self.cfg)
        self.stat_requester = StatRequester(self.cfg)
        self.data_pusher = DataPusher(self.cfg)

    def _eval_actions(self, obs, due):
        # doing inference
        actions = [None] * self.agent_num
        for i in range(self.agent_num):
            if due[i]:
                # self.last_state_action_home[i] stores the last observation, lstm state and the action of agent i
                # once the simulation of the step is complete, this will be combined with the next observation
                # and rewards then put into the trajectory buffer for agent i
                # the self.last_state_action_away[i] contains the last observation and lstm states of the enemy
                # (for value network)
                self.last_state_action_home[i] = {
                    'agent_no': i,
                    # lstm state before forward
                    'prev_state': self.lstm_states_cpu[i][0],
                    'have_teacher': self.use_teacher_model,
                    'teacher_prev_state': self.teacher_lstm_states_cpu[i][0]
                }
                self.last_state_action_home[i].update(obs[i])
                if self.agent_num == 2:
                    self.last_state_action_away[1 - i] = {
                        'agent_no': 1 - i,
                        # lstm state before forward, [0] for batch dim
                        'prev_state': self.lstm_states_cpu[1 - i][0],
                        'have_teacher': self.use_teacher_model,
                        'teacher_prev_state': self.teacher_lstm_states_cpu[1 - i][0]
                    }
                    self.last_state_action_away[1 - i].update(obs[1 - i])
                obs_copy = copy.deepcopy(obs)
                obs_copy[i] = unsqueeze_batch_dim(obs_copy[i])
                if self.cfg.env.use_cuda:
                    obs_copy[i] = to_device(obs_copy[i], 'cuda')
                if self.use_teacher_model:
                    teacher_action, teacher_logits, self.teacher_lstm_states[i] = self.teacher_agent.compute_action(
                        obs_copy[i],
                        mode="evaluate",
                        prev_states=self.teacher_lstm_states[i],
                        require_grad=False,
                        temperature=self.cfg.env.temperature
                    )
                else:
                    teacher_action = None
                    teacher_logits = None
                action, logits, self.lstm_states[i] = self.agents[i].compute_action(
                    obs_copy[i],
                    mode="evaluate",
                    prev_states=self.lstm_states[i],
                    require_grad=False,
                    temperature=self.cfg.env.temperature
                )

                if self.cfg.env.use_cuda:
                    action = to_device(action, 'cpu')
                    logits = to_device(logits, 'cpu')
                    teacher_action = to_device(teacher_action, 'cpu')
                    teacher_logits = to_device(teacher_logits, 'cpu')
                    # Two copies of next_state is maintained, one in cpu and one still in gpu
                    # TODO: is this really necessary?
                    self.lstm_states_cpu[i] = to_device(self.lstm_states[i], 'cpu')
                    self.teacher_lstm_states_cpu[i] = to_device(self.teacher_lstm_states[i], 'cpu')
                else:
                    self.lstm_states_cpu[i] = self.lstm_states[i]
                    self.teacher_lstm_states_cpu[i] = self.teacher_lstm_states[i]
                action = dict_list2list_dict(action)[0]  # 0 for batch dim
                logits = dict_list2list_dict(logits)[0]
                if self.use_teacher_model:
                    teacher_action = dict_list2list_dict(teacher_action)[0]
                    teacher_logits = dict_list2list_dict(teacher_logits)[0]
                actions[i] = action
                update_after_eval_home = {
                    # TODO: why should not this named action rather than actionS
                    'actions': action,
                    'behaviour_outputs': logits,
                    'teacher_actions': teacher_action,
                    'teacher_outputs': teacher_logits,
                    # LSTM state after forward
                    'next_state': self.lstm_states_cpu[i][0],
                    'teacher_next_state': self.teacher_lstm_states_cpu[i][0]
                }
                self.last_state_action_home[i].update(update_after_eval_home)
        if self.agent_num == 2:
            for i in range(self.agent_num):
                update_after_eval_away = {
                    # LSTM state after forward
                    'next_state': self.lstm_states_cpu[1 - i],
                    'teacher_next_state': self.teacher_lstm_states_cpu[1 - i]
                }
                self.last_state_action_away[1 - i].update(update_after_eval_away)
        return actions

    def _init_states(self):
        self.last_state_action_home = [None] * self.agent_num
        self.last_state_action_away = [None] * self.agent_num
        self.lstm_states = [[None] for i in range(self.agent_num)]
        self.lstm_states_cpu = [[None] for i in range(self.agent_num)]
        self.teacher_lstm_states = [[None] for i in range(self.agent_num)]
        self.teacher_lstm_states_cpu = [[None] for i in range(self.agent_num)]

    def run_episode(self):
        """
        Run simulation for one game episode after pulling the job using job_getter
        Load models and stats(z) using model_loader and stat_requester.
        Then pushing data to ceph, send the metadata to manager (and then forwarded to learner) using data_pusher
        Actor Logic:
            When a agent is due to act at game_step, it will take the obs and decide what action to do (after env delay)
            and when (after how many steps) should the agent be notified of newer obs and asked to act again
            this is done by calculating a delay (Note: different from env delay), and the game will proceed until
            game_step arrived the time to observe and act as requested by any of the agents.
            After every step, due[i] is set to True when steps of simulation requested by agent[i] has been completed
            and then, agent[i] need to take its action at the next step.
            Agent j with due[j]==False will be skipped and its action is None
            It's possible that two actors are simutanously required to act
            but any(due) must be True, since the simulation should keep going before reaching any requested observation
            At the beginning of the game, every agent should act and give a delay
        """
        job = self.job_getter.get_job(self.actor_uid)
        self._init_with_job(job)
        for i in range(self.agent_num):
            self.model_loader.load_model(job, i, self.agents[i].get_model())
        if self.use_teacher_model:
            self.model_loader.load_teacher_model(job, self.teacher_agent.get_model())
        if self.cfg.env.use_stat:
            for i in range(self.agent_num):
                stat = self.stat_requester.request_stat(job, i)
                if isinstance(stat, dict):
                    self.env.load_stat(stat, i)
        obs = self.env.reset()
        data_buffer = [[] for i in range(self.agent_num)]
        last_buffer = [[] for i in range(self.agent_num)]
        due = [True] * self.agent_num
        game_step = 0
        game_seconds = 0
        self._init_states()
        # main loop
        while True:
            # inferencing using the model
            actions = self._eval_actions(obs, due)
            actions = self.action_modifier(actions, game_step)
            # stepping
            game_step, due, obs, rewards, done, this_game_stat, info = self.env.step(actions)
            # assuming 22 step per second, round to integer
            # TODO:need to check with https://github.com/deepmind/pysc2/blob/master/docs/environment.md#game-speed
            game_seconds = game_step // 22
            if game_step >= self.cfg.env.game_steps_per_episode:
                # game time out, force the done flag to True
                done = True
            for i in range(self.agent_num):
                # flag telling that we should push the data buffer for agent i before next step
                at_traj_end = len(data_buffer[i]) + 1 * due[i] >= job['data_push_length'] or done
                if due[i]:
                    # we received outcome from the env, add to rollout trajectory
                    # the 'next_obs' is saved (and to be sent) if only this is the last obs of the traj
                    step_data_update_home = {
                        # the z used for the behavior network
                        'target_z': self.env.loaded_eval_stat.get_z(i),
                        # statistics calculated for this episode so far
                        'agent_z': this_game_stat[i],
                        'step': game_step,
                        'game_seconds': game_seconds,
                        'done': done,
                        'rewards': torch.tensor([rewards[i]]),
                        'info': info
                    }
                    home_step_data = merge_two_dicts(self.last_state_action_home[i], step_data_update_home)
                    if self.agent_num == 2:
                        step_data_update_away = {
                            'target_z': self.env.loaded_eval_stat.get_z(i),
                            'agent_z': this_game_stat[1 - i],
                            'step': game_step,
                            'game_seconds': game_seconds,
                            'done': done,
                            'rewards': torch.tensor([rewards[1 - i]]),
                            'info': info
                        }
                        away_step_data = merge_two_dicts(self.last_state_action_away[i], step_data_update_away)
                    else:
                        away_step_data = None
                    home_next_step_obs = obs[i] if at_traj_end else None
                    away_next_step_obs = obs[1 - i] if self.agent_num == 2 and at_traj_end else None
                    data_buffer[i].append(
                        self.compressor(
                            {
                                'home': home_step_data,
                                'away': away_step_data,
                                'home_next': home_next_step_obs,
                                'away_next': away_next_step_obs
                            }
                        )
                    )
                if at_traj_end:
                    # trajectory buffer is full or the game is finished
                    metadata = {
                        'job_id': job['job_id'],
                        'agent_no': i,
                        'agent_model_id': job['model_id'][i],
                        'job': job,
                        'step_data_compressor': self.compressor_name,
                        'game_step': game_step,
                        'done': done,
                        'finish_time': time.time(),
                        'actor_uid': self.actor_uid,
                        'info': info,
                        'traj_length': len(data_buffer[i]),  # this is the real length, without reused last traj
                        # TODO: implement other priority initialization algo, setting it to a big num now
                        'priority': 1e7,
                    }
                    if done:
                        metadata['final_reward'] = rewards[i]
                    delta = job['data_push_length'] - len(data_buffer[i])
                    # if the the data actually in the buffer when the episode ends is shorter than
                    # job['data_push_length'], the buffer is than filled with data from the last trajectory
                    if delta > 0:
                        data_buffer[i] = last_buffer[i][-delta:] + data_buffer[i]
                    metadata['length'] = len(data_buffer[i])  # should always agree with job['data_push_length']
                    self.data_pusher.push(metadata, data_buffer[i])
                    last_buffer[i] = data_buffer[i].copy()
                    data_buffer[i] = []
            if done:
                self.data_pusher.finish_job(job['job_id'])
                break

    def run(self):
        while True:
            self.run_episode()

    def action_modifier(self, actions, game_step):
        # to be overrided
        # called before actions are sent to the env, APM limits can be implemented here
        return actions

    def save_replay(self, path):
        if path:
            self.env.save_replay(path)


# The following is only a reference, never meant to be called
class JobGetter:
    def __init__(self, cfg):
        pass

    def get_job(self, actor_uid):
        """
        Overview: asking for a job from some one
        Input:
            - actor_uid
        Output:
            - job: a dict with description of how the game should be
        """
        pass


class ModelLoader:
    def __init__(self, cfg):
        pass

    def load_model(self, job, agent_no, model):
        """
        Overview: fetch a model from somewhere
        Input:
            - job: a dict with description of how the game should be
            - agent_no: 0 or 1, labeling the two agents of game
            - model: the model in agent, should be modified here
        """
        pass

    def load_teacher_model(self, job, model):
        pass


class StatRequester:
    def __init__(self, cfg):
        pass

    def request_stat(self, job, agent_no):
        pass


class DataPusher:
    def __init__(self, cfg):
        pass

    def push(self, metadata, data_buffer):
        pass

    def finish_job(self, job_id):
        pass
