import time
import copy
import pickle

import torch

import pysc2.env.sc2_env as sc2_env
from sc2learner.agent.alphastar_agent import AlphaStarAgent
from sc2learner.envs import AlphaStarEnv
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
        self.cfg.env.game_type = job['game_type']
        self.cfg.env.player1 = job['player1']
        self.cfg.env.player2 = job['player2']

        if job['game_type'] == 'game_vs_bot':
            self.agent_num = 1
            self.agents = [
                AlphaStarAgent(
                    model_config=self.cfg.model,
                    num_concurrent_episodes=1,
                    use_cuda=self.cfg.actor.use_cuda,
                    use_distributed=False
                )
            ]
        elif job['game_type'] == 'game_vs_agent':
            self.agent_num = 1
            self.agents = [
                AlphaStarAgent(
                    model_config=self.cfg.model,
                    num_concurrent_episodes=1,
                    use_cuda=self.cfg.env.use_cuda,
                    use_distributed=False
                )
            ]
        elif job['game_type'] == 'agent_vs_agent':
            self.agent_num = 2
            self.agents = [
                AlphaStarAgent(
                    model_config=self.cfg.model,
                    num_concurrent_episodes=1,
                    use_cuda=self.cfg.actor.use_cuda,
                    use_distributed=False
                ),
                AlphaStarAgent(
                    model_config=self.cfg.model,
                    num_concurrent_episodes=1,
                    use_cuda=self.cfg.actor.use_cuda,
                    use_distributed=False
                )
            ]
        else:
            raise NotImplementedError()

        for agent in self.agents:
            agent.train()
            agent.set_seed(job['random_seed'])
            agent.reset_previous_state([True])  # Here the lstm_states are reset

        if job['teacher_model_id']:
            # agent for evaluation of the SL model to produce the teacher_logits
            # for human_policy_kl_loss in rl.py of AlphaStar Supp. Mat.
            self.teacher_agent = AlphaStarAgent(
                model_config=self.cfg.model,
                num_concurrent_episodes=1,
                use_cuda=self.cfg.actor.use_cuda,
                use_distributed=False
            )
            self.teacher_agent.eval()
            self.teacher_agent.set_seed(job['random_seed'])
            self.teacher_agent.reset_previous_state([True])
            self.use_teacher_model = True
        else:
            self.use_teacher_model = False
        self.env = self._make_env()
        self.compressor_name = job['step_data_compressor']
        self.compressor = get_step_data_compressor(self.compressor_name)

    def _make_env(self):
        return AlphaStarEnv(self.cfg)

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
                    self.last_state_action_away[i] = {
                        'agent_no': 1 - i,
                        # lstm state before forward, [0] for batch dim
                        'prev_state': self.lstm_states_cpu[1 - i][0],
                        'have_teacher': self.use_teacher_model,
                        'teacher_prev_state': self.teacher_lstm_states_cpu[1 - i][0]
                    }
                    self.last_state_action_away[i].update(obs[1 - i])
                obs_copy = copy.deepcopy(obs)
                obs_copy[i] = unsqueeze_batch_dim(obs_copy[i])
                if self.cfg.actor.use_cuda:
                    obs_copy[i] = to_device(obs_copy[i], 'cuda')
                if self.use_teacher_model:
                    teacher_action, teacher_logits, self.teacher_lstm_states[i] = self.teacher_agent.compute_action(
                        obs_copy[i],
                        mode="evaluate",
                        prev_states=self.teacher_lstm_states[i],
                        require_grad=False,
                        temperature=self.cfg.train.temperature
                    )
                else:
                    teacher_action = None
                    teacher_logits = None
                action, logits, self.lstm_states[i] = self.agents[i].compute_action(
                    obs_copy[i],
                    mode="evaluate",
                    prev_states=self.lstm_states[i],
                    require_grad=False,
                    temperature=self.cfg.train.temperature
                )

                if self.cfg.actor.use_cuda:
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
                # remove evaluate related key
                send_action = copy.deepcopy(action)
                if 'action_entity_raw' in send_action:
                    send_action.pop('action_entity_raw')
                send_teacher_action = copy.deepcopy(teacher_action)
                if 'action_entity_raw' in send_teacher_action:
                    send_teacher_action.pop('action_entity_raw')
                # correct action selected_units
                send_action = self._correct_send_action(send_action, logits)
                update_after_eval_home = {
                    'actions': send_action,
                    'behaviour_outputs': logits,
                    'teacher_actions': send_teacher_action,
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
                self.last_state_action_away[i].update(update_after_eval_away)
        return actions

    def _correct_send_action(self, send_action, outputs):
        action_su = send_action['selected_units']
        outputs_su = outputs['selected_units']
        if action_su is not None:
            if action_su.shape[0] == outputs_su.shape[0] - 1:
                # add end_flag label
                device = action_su.device
                end_flag_label = torch.LongTensor([outputs_su.shape[1] - 1]).to(device)
                send_action['selected_units'] = torch.cat([action_su, end_flag_label], dim=0)
        return send_action

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
            When a agent is due to act at game_loop, it will take the obs and decide what action to do (after env delay)
            and when (after how many steps) should the agent be notified of newer obs and asked to act again
            this is done by calculating a delay (Note: different from env delay), and the game will proceed until
            game_loop arrived the time to observe and act as requested by any of the agents.
            After every step, due[i] is set to True when steps of simulation requested by agent[i] has been completed
            and then, agent[i] need to take its action at the next step.
            Agent j with due[j]==False will be skipped and its action is None
            It's possible that two actors are simutanously required to act
            but any(due) must be True, since the simulation should keep going before reaching any requested observation
            At the beginning of the game, every agent should act and give a delay
        """
        job = self.job_getter.get_job(self.actor_uid)
        self._init_with_job(job)
        # load model
        for i in range(self.agent_num):
            self.model_loader.load_model(job, i, self.agents[i].get_model())
        # teacher_model is fixed in the whole RL training
        if self.use_teacher_model:
            self.model_loader.load_teacher_model(job, self.teacher_agent.get_model())
        # load stat
        stat = [self.stat_requester.request_stat(job, i) for i in range(self.agent_num)]
        assert all(isinstance(t, dict) for t in stat)
        # reset
        obs = self.env.reset(stat)
        self._init_states()
        # initialize loop variable
        data_buffer = [[] for i in range(self.agent_num)]
        last_buffer = [[] for i in range(self.agent_num)]  # for non-aligned trajectory length
        due = [True] * self.agent_num
        game_loop = 0
        game_seconds = 0
        trajectory_count = 0
        # main loop
        while True:
            # inferencing using the model
            actions = self._eval_actions(obs, due)
            actions = self.action_modifier(actions, game_loop)
            # stepping
            game_loop, due, obs, rewards, done, info = self.env.step(actions)
            game_loop = int(game_loop)  # np.int32->int
            # assuming 22 step per second, round to integer
            game_seconds = game_loop // 22
            if game_loop >= self.cfg.env.game_steps_per_episode:
                # game time out, force the done flag to True
                done = True
            if len(data_buffer[0]) % self.cfg.actor.print_freq == 0:
                print(
                    'actor: uid({}), game_loop({}), len of data_buffer({})'.format(
                        self.actor_uid, game_loop, len(data_buffer[0])
                    )
                )
            for i in range(self.agent_num):
                # flag telling that we should push the data buffer for agent i before next step
                at_traj_end = len(data_buffer[i]) + 1 * due[i] >= job['data_push_length'] or done
                if due[i]:
                    # we received outcome from the env, add to rollout trajectory
                    # the 'next_obs' is saved (and to be sent) if only this is the last obs of the traj
                    step_data_update_home = {
                        'step': game_loop,
                        'game_seconds': game_seconds,
                        'done': done,
                        'rewards': rewards[i],
                        #'info': info
                    }
                    home_step_data = merge_two_dicts(self.last_state_action_home[i], step_data_update_home)
                    if self.agent_num == 2:
                        step_data_update_away = {
                            'step': game_loop,
                            'game_seconds': game_seconds,
                            'done': done,
                            'rewards': rewards[1 - i],
                            #'info': info
                        }
                        away_step_data = merge_two_dicts(self.last_state_action_away[i], step_data_update_away)
                    else:
                        away_step_data = None
                    step_data = {'home': home_step_data, 'away': away_step_data}
                    if at_traj_end:
                        step_data['home_next'] = obs[i]
                    if self.agent_num == 2 and at_traj_end:
                        step_data['away_next'] = obs[1 - i]
                    data_buffer[i].append(self.compressor(step_data))
                if at_traj_end:
                    # trajectory buffer is full or the game is finished
                    metadata = {
                        'job_id': job['job_id'],
                        'agent_no': i,
                        'agent_model_id': job['model_id'][i],
                        'job': job,
                        'step_data_compressor': self.compressor_name,
                        'game_loop': game_loop,
                        'done': done,
                        'finish_time': time.time(),
                        'actor_uid': self.actor_uid,
                        #'info': info,
                        'traj_length': len(data_buffer[i]),  # this is the real length, without reused last traj
                        # TODO(nyz): implement other priority initialization algo, setting it to 1.0 now
                        'priority': 1.0,
                    }
                    delta = job['data_push_length'] - len(data_buffer[i])
                    # if the the data actually in the buffer when the episode ends is shorter than
                    # job['data_push_length'], the buffer is than filled with data from the last trajectory
                    if delta > 0:
                        data_buffer[i] = last_buffer[i][-delta:] + data_buffer[i]
                    metadata['length'] = len(data_buffer[i])  # should always agree with job['data_push_length']
                    self.data_pusher.push(metadata, data_buffer[i])
                    last_buffer[i] = data_buffer[i].copy()
                    data_buffer[i] = []
                    trajectory_count += 1
                    # when several trajectories are finished, reload(update) model
                    if trajectory_count % self.cfg.actor.load_model_freq == 0:
                        for i in range(self.agent_num):
                            self.model_loader.load_model(job, i, self.agents[i].get_model())

            if done:
                result_map = {1: 'wins', 0: 'draws', -1: 'losses'}
                result = result_map[rewards[0]['winloss'].int().item()]
                self.data_pusher.finish_job(job['job_id'], result)
                break

    def run(self):
        while True:
            self.run_episode()

    def action_modifier(self, actions, game_loop):
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

    def finish_job(self, job_id, result):
        pass
