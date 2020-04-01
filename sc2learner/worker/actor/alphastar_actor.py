import time
import copy
import torch

import pysc2.env.sc2_env as sc2_env
from sc2learner.agent.alphastar_agent import AlphaStarAgent
from sc2learner.envs.alphastar_env import AlphaStarEnv
from sc2learner.utils import get_actor_uid, dict_list2list_dict, merge_two_dicts
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
    """
    AlphaStar Actor
    Contents of each entry in the trajectory dict
        - 'step'
        - 'agent_no'
        - 'prev_obs'
        - 'lstm_state_before': LSTM state before the step
        - 'lstm_state_after': LSTM state after the step
        - 'logits': action logits
        - 'action'
        - 'next_obs'
        - 'done'
        - 'rewards'
        - 'info'
    """
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

    def _init_with_job(self, job):
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
            # agent for evaluation of the SL model to produce the logits
            # for human_policy_kl_loss in rl.py of AlphaStar Supp. Mat.
            self.teacher_agent = AlphaStarAgent(
                model_config=self.cfg.model,
                num_concurrent_episodes=1,
                use_cuda=self.cfg.env.use_cuda,
                use_distributed=False
            )
            self.use_teacher_model = True
        else:
            self.use_teacher_model = False
        self.env = self._make_env(players)

    def _make_env(self, players):
        return AlphaStarEnv(self.cfg, players)

    # this is to be overridden
    def _module_init(self):
        self.job_getter = JobGetter(self.cfg)
        self.model_loader = ModelLoader(self.cfg)
        self.stat_requester = StatRequester(self.cfg)
        self.data_pusher = DataPusher(self.cfg)

    def _init_states(self):
        self.last_state_action = [None] * self.agent_num
        self.lstm_states = [None] * self.agent_num
        self.lstm_states_cpu = [None] * self.agent_num
        self.teacher_lstm_states = [None] * self.agent_num
        self.teacher_lstm_states_cpu = [None] * self.agent_num

    def _eval_actions(self, obs, due):
        actions = [None] * self.agent_num
        for i in range(self.agent_num):
            if due[i]:
                # self.last_state_action[i] stores the last observation, lstm state and the action of agent i
                # once the simulation of the step is complete, this will be combined with the next observation
                # and rewards then put into the trajectory buffer for agent i
                self.last_state_action[i] = {
                    'agent_no': i,
                    'prev_obs': obs,
                    'lstm_state_before': self.lstm_states_cpu[i],
                    'have_teacher': self.use_teacher_model,
                    'teacher_lstm_state_before': self.teacher_lstm_states_cpu[i]
                }
                obs_copy = copy.deepcopy(obs)
                obs_copy[i] = unsqueeze_batch_dim(obs_copy[i])
                if self.cfg.env.use_cuda:
                    obs_copy[i] = to_device(obs_copy[i], 'cuda')
                if self.use_teacher_model:
                    _, teacher_logits, self.teacher_lstm_states[i] = self.teacher_agent.compute_action(
                        obs_copy[i],
                        mode="evaluate",
                        prev_states=self.teacher_lstm_states[i],
                        require_grad=False,
                        temperature=self.cfg.env.temperature
                    )
                else:
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
                    teacher_logits = to_device(teacher_logits, 'cpu')
                    # Two copies of next_state is maintained, one in cpu and one still in gpu
                    # TODO: is this really necessary?
                    self.lstm_states_cpu[i] = to_device(self.lstm_states[i], 'cpu')
                    self.teacher_lstm_states_cpu[i] = to_device(self.teacher_lstm_states[i], 'cpu')
                else:
                    self.lstm_states_cpu[i] = self.lstm_states[i]
                    self.teacher_lstm_states_cpu[i] = self.teacher_lstm_states[i]
                action = dict_list2list_dict(action)[0]  # 0 for batch dim

                actions[i] = action
                update_after_eval = {
                    'action': action,
                    'logits': logits,
                    'teacher_logits': teacher_logits,
                    'lstm_state_after': self.lstm_states_cpu[i],
                    'teacher_lstm_state_after': self.teacher_lstm_states_cpu[i]
                }
                self.last_state_action[i] = merge_two_dicts(self.last_state_action[i], update_after_eval)
        return actions

    def run_episode(self):
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
        data_buffer = [[]] * self.agent_num
        # Actor Logic:
        # When a agent is due to act at game_step, it will take the obs and decide what action to do (after env delay)
        # and when (after how many steps) should the agent be notified of newer obs and asked to act again
        # this is done by calculating a delay (Note:different from env delay), and the game will proceed until
        # game_step is at the time to obs and act requested by any of the agents.
        # due[i] is set to True when agent[i] requested steps of simulation has been completed
        # and then, agent[i] need to take its action at the next step.
        # Agent j with due[j]==False will be skipped and its action is None
        # It's possible that two actors are simutanously required to act
        # but any(due) must be True, since the simulation should keep going before reaching any requested observation
        # At the beginning of the game, every agent should act and give a delay
        due = [True] * self.agent_num
        game_step = 0
        self._init_states()
        # main loop
        while True:
            actions = self._eval_actions(obs, due)
            actions = self.action_modifier(actions, game_step)
            game_step, due, obs, rewards, done, info = self.env.step(actions)
            if game_step >= self.cfg.env.game_steps_per_episode:
                # game time out, force the done flag to True
                done = True
            for i in range(self.agent_num):
                if due[i]:
                    # we received obs from the env, add to rollout trajectory
                    obs_data = {'step': game_step, 'next_obs': obs, 'done': done, 'rewards': rewards[i], 'info': info}
                    data_buffer[i].append(merge_two_dicts(self.last_state_action[i], obs_data))
                if len(data_buffer[i]) >= job['data_push_length'] or done:
                    # trajectory buffer is full or the game is finished
                    # so the length of a trajectory may not necessary be data_push_length
                    metadata = {
                        'job_id': job['job_id'],
                        'agent_no': i,
                        'agent_model_id': job['model_id'][i],
                        'job': job,
                        'game_step': game_step,
                        'done': done,
                        'finish_time': time.time(),
                        'actor_uid': self.actor_uid,
                        'info': info,
                        'traj_length': len(data_buffer[i]),
                    }
                    if done:
                        metadata['final_reward'] = rewards[i]
                    self.data_pusher.push(metadata, data_buffer[i])
                    data_buffer[i] = []
            if done:
                break

    def run(self):
        while True:
            self.run_episode()

    def action_modifier(self, actions, game_step):
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
