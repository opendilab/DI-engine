import numpy as np
import torch
import copy
import pysc2.env.sc2_env as sc2_env
from pysc2.env.sc2_env import SC2Env
from pysc2.lib.actions import FunctionCall, FUNCTIONS, RAW_FUNCTIONS
from pysc2.lib.static_data import NUM_ACTIONS, ACTIONS_REORDER_INV
from sc2learner.envs.observations.alphastar_obs_wrapper import SpatialObsWrapper, ScalarObsWrapper, EntityObsWrapper,\
    transform_spatial_data, transform_scalar_data, transform_entity_data
from sc2learner.envs.actions.alphastar_act_wrapper import AlphastarActParser
from sc2learner.envs import get_available_actions_processed_data


class AlphastarEnv(SC2Env):
    def __init__(self, cfg, players):
        """
        Input:
            - cfg
            - players:list of two sc2_env.Agent or sc2_env.Bot in the game
        """
        agent_interface_format = sc2_env.parse_agent_interface_format(
            feature_screen=cfg.env.screen_resolution, feature_minimap=cfg.env.map_size
        )
        self.map_size = cfg.env.map_size
        self.agent_num = sum([isinstance(p, sc2_env.Agent) for p in players])
        assert (self.agent_num <= 2)
        super(AlphastarEnv, self).__init__(
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
        self.spatial_wrapper = SpatialObsWrapper(transform_spatial_data())
        self.entity_wrapper = EntityObsWrapper(transform_entity_data())
        template_obs, template_replay, template_act = transform_scalar_data()
        self.scalar_wrapper = ScalarObsWrapper(template_obs)
        self.template_act = template_act
        self.action_num = NUM_ACTIONS
        self.use_global_cumulative_stat = cfg.env.use_global_cumulative_stat
        self.use_available_action_transform = cfg.env.use_available_action_transform

        self.use_stat = cfg.env.use_stat
        if self.use_stat:
            self.stat = self._init_stat(cfg.env.stat_path, cfg.env.beginning_build_order_num)
        self._reset_flag = False

    def _init_stat(self, path, begin_num):
        stat = torch.load(path)
        stat['beginning_build_order'] = stat['beginning_build_order'][:begin_num]
        if stat['beginning_build_order'].shape[0] < begin_num:
            B, N = stat['beginning_build_order'].shape
            B0 = begin_num - B
            stat['beginning_build_order'] = torch.cat([stat['beginning_build_order'], torch.zeros(B0, N)])
        return stat

    def _merge_stat(self, obs):
        obs['scalar_info']['mmr'] = self.stat['mmr']
        obs['scalar_info']['beginning_build_order'] = self.stat['beginning_build_order']
        if self.use_global_cumulative_stat:
            obs['scalar_info']['cumulative_stat'] = self.stat['cumulative_stat']
        return obs

    def _merge_action(self, obs, last_action):
        if isinstance(last_action['action_type'], torch.Tensor):
            for index, item in enumerate(last_action['action_type']):
                last_action['action_type'][index] = ACTIONS_REORDER_INV[item.item()]

        last_action_type = last_action['action_type']
        last_delay = last_action['delay']
        last_queued = last_action['queued']
        last_queued = last_queued if isinstance(last_queued, torch.Tensor) else torch.LongTensor([2])  # 2 as 'none'
        obs['scalar_info']['last_delay'] = self.template_act[0]['op'](torch.LongTensor([last_delay])).squeeze()
        obs['scalar_info']['last_queued'] = self.template_act[1]['op'](torch.LongTensor([last_queued])).squeeze()
        obs['scalar_info']['last_action_type'] = self.template_act[2]['op'](torch.LongTensor([last_action_type])
                                                                            ).squeeze()

        N = obs['entity_info'].shape[0]
        if obs['entity_info'] is None:
            obs['entity_info'] = torch.cat([obs['entity_info'], torch.zeros(N, 4)], dim=1)
            return obs
        selected_units = last_action['selected_units']
        target_units = last_action['target_units']
        obs['entity_info'] = torch.cat([obs['entity_info'], torch.zeros(N, 2)], dim=1)
        selected_units = selected_units if isinstance(selected_units, torch.Tensor) else []
        for idx, v in enumerate(obs['entity_raw']['id']):
            if v in selected_units:
                obs['entity_info'][idx, -1] = 1
            else:
                obs['entity_info'][idx, -2] = 1

        obs['entity_info'] = torch.cat([obs['entity_info'], torch.zeros(N, 2)], dim=1)
        target_units = target_units if isinstance(target_units, torch.Tensor) else []
        for idx, v in enumerate(obs['entity_raw']['id']):
            if v in target_units:
                obs['entity_info'][idx, -1] = 1
            else:
                obs['entity_info'][idx, -2] = 1
        return obs

    def _get_obs(self, obs, last_actions):
        if 'enemy_upgrades' not in obs.keys():
            obs['enemy_upgrades'] = np.array([0])
        entity_info, entity_raw = self.entity_wrapper.parse(obs)
        new_obs = {
            'scalar_info': self.scalar_wrapper.parse(obs),
            'spatial_info': self.spatial_wrapper.parse(obs),
            'entity_info': entity_info,
            'entity_raw': entity_raw,
            'map_size': [self.map_size[1], self.map_size[0]],  # x,y -> y,x
        }

        new_obs = self._merge_action(new_obs, last_actions)
        if self.use_stat:
            new_obs = self._merge_stat(new_obs)
        if self.use_available_action_transform:
            new_obs = get_available_actions_processed_data(new_obs)
        return new_obs

    def _get_action(self, actions):
        action_type = actions['action_type']
        delay = actions['delay']
        # action target location transform
        target_location = actions['target_location']
        if target_location is not None:
            x = target_location[1]
            y = target_location[0]
            y = self.map_size[0] - y
            actions['target_location'] = [x, y]

        arg_keys = ['queued', 'selected_units', 'target_units', 'target_location']
        args = [v for k, v in actions.items() if k in arg_keys and v is not None]
        return FunctionCall.init_with_validation(action_type, args, raw=True), delay

    def step(self, actions):
        """
        Overview: Apply actions, step the world forward, and return observations.
        Input:
            - actions: list of actions for each agent, length should be the number of agents
            if an agent don't want to act this time, the action should be set to None
        Return:
            - step: actually how many steps are forwarded during this call
            - due: list of bool telling which agents should take the next action
            - obs: list of ob dicts for two agents after taking action
            - rewards: win/loss reward
            - done: if the game terminated
            - info
        """
        assert (self._reset_flag)
        transformed_actions = [None] * self.agent_num
        for n in range(self.agent_num):
            action = actions[n]
            if action is not None:
                t, d = self._get_action(action)
                transformed_actions[n] = t
                self._next_obs[n] = self._episode_steps + d
                self.last_actions[n] = action
            else:
                transformed_actions[n] = []
        step_mul = min(self._next_obs) - self._episode_steps
        if step_mul == 0:
            # repeat last observation and store last action
            # at least one agent requested by returning zero delay
            for n in range(self.agent_num):
                if transformed_actions[n]:
                    self._buffered_actions[n].append(transformed_actions[n])
            _, _, obs, rewards, done, info = self._last_output
            due = [s <= self._episode_steps for s in self._next_obs]
        else:
            for n in range(self.agent_num):
                if transformed_actions[n]:
                    transformed_actions[n] = self._buffered_actions[n] + [transformed_actions[n]]
            assert (any(transformed_actions))
            assert (step_mul >= 0), 'Some agent requested negative delay!'
            timesteps = super().step(transformed_actions, step_mul=step_mul)
            due = [s <= self._episode_steps for s in self._next_obs]
            assert (any(due))
            self._buffered_actions = [[]] * self.agent_num
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
                    obs[n] = self._get_obs(o, self.last_actions[n])
            self._last_output = [step_mul, due, obs, rewards, done, info]
        # as obs may be changed somewhere in parsing
        # we have to return a copy to keep the self._last_ouput intact
        return step_mul, due, copy.deepcopy(obs), rewards, done, info

    def reset(self):
        timesteps = super().reset()
        last_action = {
            'action_type': 0,
            'delay': 0,
            'queued': None,
            'selected_units': None,
            'target_units': None,
            'target_location': None
        }
        self.last_actions = [last_action] * self.agent_num
        obs = [self._get_obs(timestep.observation, last_action) for timestep in timesteps]
        infos = [timestep.game_info for timestep in timesteps]
        self.map_size = infos[0].start_raw.map_size
        self.map_size = (self.map_size.x, self.map_size.y)
        self._next_obs = [0] * self.agent_num
        self._episode_steps = 0
        self._reset_flag = True
        self._buffered_actions = [[]] * self.agent_num
        self._last_output = [0, [True] * self.agent_num, obs, [0] * self.agent_num, False, infos]
        return copy.deepcopy(obs)
