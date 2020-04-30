import copy

import numpy as np
import torch

import pysc2.env.sc2_env as sc2_env
from pysc2.env.sc2_env import SC2Env
from pysc2.lib.actions import FunctionCall
from pysc2.lib.static_data import NUM_ACTIONS, ACTIONS_REORDER_INV
from sc2learner.envs import get_available_actions_processed_data, get_map_size, get_enemy_upgrades_processed_data
from sc2learner.envs.observations.alphastar_obs_wrapper import SpatialObsWrapper, ScalarObsWrapper, EntityObsWrapper, \
    transform_spatial_data, transform_scalar_data, transform_entity_data
from sc2learner.envs.statistics import Statistics, GameLoopStatistics


class AlphaStarEnv(SC2Env):
    def __init__(self, cfg, players):
        """
        Input:
            - cfg
            - players:list of two sc2_env.Agent or sc2_env.Bot in the game
        """
        self.cfg = cfg
        self.map_size = get_map_size(cfg.env.map_name, cropped=cfg.env.crop_map_to_playable_area)

        agent_interface_format = sc2_env.parse_agent_interface_format(
            feature_screen=cfg.env.screen_resolution,
            feature_minimap=self.map_size,  # x, y
            raw_crop_to_playable_area=cfg.env.crop_map_to_playable_area,
            action_delays=cfg.env.get('action_delays')
        )

        self.agent_num = sum([isinstance(p, sc2_env.Agent) for p in players])
        assert (self.agent_num <= 2)
        super(AlphaStarEnv, self).__init__(
            map_name=cfg.env.map_name,
            random_seed=cfg.env.random_seed,
            step_mul=cfg.env.default_step_mul,
            players=players,
            game_steps_per_episode=cfg.env.game_steps_per_episode,
            agent_interface_format=agent_interface_format,
            disable_fog=cfg.env.disable_fog,
            score_index=-1,  # use win/loss reward rather than score
            ensure_available_actions=False,
            realtime=cfg.env.realtime,
        )
        self._spatial_wrapper = SpatialObsWrapper(transform_spatial_data())
        self._entity_wrapper = EntityObsWrapper(transform_entity_data())
        template_obs, template_replay, template_act = transform_scalar_data()
        self._scalar_wrapper = ScalarObsWrapper(template_obs)
        self._template_act = template_act
        self._use_available_action_transform = cfg.env.use_available_action_transform
        self._obs_stat_type = cfg.env.obs_stat_type
        assert self._obs_stat_type in ['replay_online', 'self_online', 'replay_last']

        self._use_stat = cfg.env.use_stat
        self._reset_flag = False
        # This is the human games statistics used as an input of network
        self.loaded_eval_stat = [None] * self.agent_num
        self.enemy_upgrades = [None] * self.agent_num

    def load_stat(self, stat, agent_no):
        """
        Set the statistics to be append to every observation of each agent
        stat: stat dict processed by transform_stat
        agent_no: 0 or 1
        """
        assert self._use_stat, 'We should not load stat when we are not going to use stat'
        begin_num = self.cfg.env.beginning_build_order_num
        self.loaded_eval_stat[agent_no] = GameLoopStatistics(stat, begin_num)
        self._episode_stat.load_global_bo(agent_no, self.loaded_eval_stat[agent_no].global_bo)

    def _merge_stat(self, obs, agent_no, game_loop=None):
        """
        Append the statistics to the observation
        """
        if self._obs_stat_type == 'replay_online':
            stat = self.loaded_eval_stat[agent_no].get_input_z_by_game_loop(game_loop)
        elif self._obs_stat_type == 'self_online':
            stat = self._episode_stat.get_transformed_stat(agent_no)
        elif self._obs_stat_type == 'replay_last':
            stat = self.loaded_eval_stat[agent_no].get_input_z_by_game_loop(None)

        assert set(stat.keys) == set(['mmr', 'beginning_build_order', 'cumulative_stat'])
        obs['scalar_info'].update(stat)
        return obs

    def _merge_action(self, obs, last_action, add_dim=True):
        """
        adding information of last action to the scalar_info in the observations
        see detailed-architecture.txt L112-L114
        and encode whether the entities are selected or targeted in last action
        see detailed-architechure.txt L61-62
        """
        last_action_type = last_action['action_type']
        last_delay = last_action['delay']
        last_queued = last_action['queued']
        last_queued = last_queued if isinstance(last_queued, torch.Tensor) else torch.LongTensor([2])  # 2 as 'none'
        obs['scalar_info']['last_delay'] = self._template_act[0]['op'](torch.LongTensor([last_delay])).squeeze()
        obs['scalar_info']['last_queued'] = self._template_act[1]['op'](torch.LongTensor([last_queued])).squeeze()
        obs['scalar_info']['last_action_type'] = self._template_act[2]['op'](torch.LongTensor([last_action_type])
                                                                             ).squeeze()

        N = obs['entity_info'].shape[0]
        if obs['entity_info'] is None:
            obs['entity_info'] = torch.cat([obs['entity_info'], torch.zeros(N, 4)], dim=1)
            return obs
        if add_dim:
            obs['entity_info'] = torch.cat([obs['entity_info'], torch.zeros(N, 4)], dim=1)
        else:
            obs['entity_info'][:, -4].zero_()

        selected_units = last_action['selected_units']
        target_units = last_action['target_units']
        selected_units = selected_units if isinstance(selected_units, torch.Tensor) else []
        for idx, v in enumerate(obs['entity_raw']['id']):
            if v in selected_units:
                obs['entity_info'][idx, -3] = 1
            else:
                obs['entity_info'][idx, -4] = 1

        target_units = target_units if isinstance(target_units, torch.Tensor) else []
        for idx, v in enumerate(obs['entity_raw']['id']):
            if v in target_units:
                obs['entity_info'][idx, -1] = 1
            else:
                obs['entity_info'][idx, -2] = 1
        return obs

    def _get_obs(self, obs, last_actions, agent_no):
        # post process observations returned from sc2env
        entity_info, entity_raw = self._entity_wrapper.parse(obs)
        new_obs = {
            'scalar_info': self._scalar_wrapper.parse(obs),
            'spatial_info': self._spatial_wrapper.parse(obs),
            'entity_info': entity_info,
            'entity_raw': entity_raw,
            'map_size': [self.map_size[1], self.map_size[0]],  # x,y -> y,x
        }

        new_obs = self._merge_action(new_obs, last_actions)
        if self._use_stat:
            new_obs = self._merge_stat(new_obs, agent_no)
        if self._use_available_action_transform:
            new_obs = get_available_actions_processed_data(new_obs)
        self.enemy_upgrades[agent_no] = get_enemy_upgrades_processed_data(new_obs, self.enemy_upgrades[agent_no])
        new_obs['scalar_info']['enemy_upgrades'] = self.enemy_upgrades[agent_no]
        return new_obs

    def _transform_action(self, action):
        # convert network output to SC2 raw input
        action = copy.deepcopy(action)
        # tensor2value
        for k, v in action.items():
            if isinstance(v, torch.Tensor):
                if k == 'action_type':
                    action[k] = ACTIONS_REORDER_INV[v.item()]
                elif k in ['selected_units', 'target_units', 'target_location']:
                    action[k] = v.tolist()
                elif k in ['queued', 'delay']:
                    action[k] = v.item()
                elif k == 'action_entity_raw':
                    pass
                else:
                    raise KeyError("invalid key:{}".format(k))
        # action unit id transform
        for k in ['selected_units', 'target_units']:
            if action[k] is not None:
                unit_ids = []
                for unit in action[k]:
                    unit_ids.append(action['action_entity_raw'][unit]['id'])
                action[k] = unit_ids
        # action target location transform
        target_location = action['target_location']
        if target_location is not None:
            x = target_location[1]
            y = target_location[0]
            y = self.map_size[1] - y
            action['target_location'] = [x, y]
        return action

    def _get_action(self, actions):
        # Convert transformed actions to pysc2 FunctionCalls
        action_type = actions['action_type']
        delay = actions['delay']
        arg_keys = ['queued', 'selected_units', 'target_units', 'target_location']
        args = [v for k, v in actions.items() if k in arg_keys and v is not None]
        return FunctionCall.init_with_validation(action_type, args, raw=True), delay

    def step(self, actions):
        """
        Overview: Apply actions, step the world forward, and return observations.
        Input:
            - actions: list of action for each agent, length should be the number of agents
            if an agent don't want to act this time, the action should be set to None
        Return:
            - step: total game steps after this call
            - due: list of bool telling which agents should take the next action
            - obs: list of ob dicts for two agents after taking action
            - rewards: win/loss reward
            - done: if the game terminated
            - stat: the statistics for what happened in the current episode, in get_z format
            - info
        """
        assert (self._reset_flag)
        sc2_actions = [None] * self.agent_num
        for n in range(self.agent_num):
            action = actions[n]
            if action is not None:
                transformed_action = self._transform_action(action)
                t, d = self._get_action(transformed_action)
                sc2_actions[n] = t
                self._next_obs[n] = self._episode_steps + d
                self.last_actions[n] = transformed_action
            else:
                sc2_actions[n] = []
        step_mul = min(self._next_obs) - self._episode_steps
        if step_mul == 0:
            # repeat last observation and store last action
            # as at least one agent requested this by returning delay=0
            for n in range(self.agent_num):
                if sc2_actions[n]:
                    self._buffered_actions[n].append(sc2_actions[n])
            _, _, obs, rewards, done, episode_stat, info = self._last_output
            for n in range(self.agent_num):
                obs[n] = self._merge_action(obs[n], self.last_actions[n], add_dim=False)
            due = [s <= self._episode_steps for s in self._next_obs]
        else:
            for n in range(self.agent_num):
                if sc2_actions[n]:
                    # append buffered actions to current actions list
                    sc2_actions[n] = self._buffered_actions[n] + [sc2_actions[n]]
            assert (any(sc2_actions))
            assert (step_mul >= 0), 'Some agent requested negative delay!'
            # Note: the SC2Env can accept actions like [[agent1_act1, agent1_act2], []]
            # but I can not simutanously do multiple actions (may need a external loop)
            timesteps = super().step(sc2_actions, step_mul=step_mul)
            due = [s <= self._episode_steps for s in self._next_obs]
            assert (any(due))
            self._buffered_actions = [[] for i in range(self.agent_num)]
            done = False
            obs = [None] * self.agent_num
            rewards = [None] * self.agent_num
            info = [None] * self.agent_num
            for n in range(self.agent_num):
                timestep = timesteps[n]
                if timestep is not None:
                    done = done or timestep.last()
                    _, rewards[n], _, o, info[n] = timestep
                    assert (self.last_actions[n])
                    obs[n] = self._get_obs(o, self.last_actions[n], n)
                if due[n]:
                    assert (self.last_actions[n])
                    self._episode_stat.update_stat(self.last_actions[n], obs[n], n)
            episode_stat = [self._episode_stat.get_z(n) for n in range(self.agent_num)]
            self._last_output = [self._episode_steps, due, obs, rewards, done, episode_stat, info]
        # as obs may be changed somewhere in parsing
        # we have to return a copy to keep the self._last_ouput intact
        return self._episode_steps, due, copy.deepcopy(obs), rewards, done, episode_stat, info

    def reset(self):
        """
        Reset the env, should be called before start stepping
        Return:
            obs: initial observation
        """
        timesteps = super().reset()
        last_action = {
            'action_type': torch.Tensor([0]),
            'delay': torch.Tensor([0]),
            'queued': None,
            'selected_units': None,
            'target_units': None,
            'target_location': None
        }
        self.last_actions = [last_action] * self.agent_num
        # This is for the statistics of current episode actions and obs
        self._episode_stat = Statistics(player_num=self.agent_num, begin_num=self.cfg.env.get('begin_num', 200))
        obs = []
        for n in range(self.agent_num):
            obs.append(self._get_obs(timesteps[n].observation, last_action, n))
        infos = [timestep.game_info for timestep in timesteps]
        env_provided_map_size = infos[0].start_raw.map_size
        env_provided_map_size = [env_provided_map_size.x, env_provided_map_size.y]
        assert tuple(env_provided_map_size) == tuple(self.map_size), \
            "Environment uses a different map size {} compared to config " \
            "{}.".format(env_provided_map_size, self.map_size)

        self._next_obs = [0] * self.agent_num
        self._episode_steps = 0
        self._reset_flag = True
        self._buffered_actions = [[] for i in range(self.agent_num)]
        self._last_output = [
            0, [True] * self.agent_num, obs, [0] * self.agent_num, False,
            [self._episode_stat.get_z(n) for n in range(self.agent_num)], infos
        ]
        return copy.deepcopy(obs)

    def transformed_action_to_string(self, action):
        # producing human readable debug output
        return '[Action: type({}) delay({}) queued({}) selected_units({}) target_units({}) target_location({})]'.format(
            action['action_type'], action['delay'], action['queued'], action['selected_units'], action['target_units'],
            action['target_location']
        )

    def action_to_string(self, action):
        # producing human readable debug output from network output
        if action is None:
            return 'None'
        action = self._transform_action(action)
        return self.transformed_action_to_string(action)

    def get_action_type(self, action):
        # get transformed action type from network output
        if action is None:
            return None
        action = self._transform_action(action)
        return action['action_type']

    def get_target_z(self, agent_no, game_loop):
        return self.loaded_eval_stat[agent_no].get_reward_z_by_game_loop(game_loop)
