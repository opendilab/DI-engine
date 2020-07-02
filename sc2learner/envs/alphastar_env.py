import copy
from collections import OrderedDict, namedtuple

import numpy as np
import torch
import torch.nn.functional as F

import pysc2.env.sc2_env as sc2_env
from pysc2.env.sc2_env import SC2Env
from pysc2.lib.actions import FunctionCall
from pysc2.lib.static_data import NUM_ACTIONS, ACTIONS_REORDER_INV
from sc2learner.envs import get_available_actions_processed_data, get_map_size, get_enemy_upgrades_processed_data
from .observations.alphastar_obs_wrapper import SpatialObsWrapper, ScalarObsWrapper, EntityObsWrapper, \
    transform_spatial_data, transform_scalar_data, transform_entity_data, clip_one_hot
from .statistics import GameLoopStatistics, RealTimeStatistics
from .rewards import RewardHelper


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
            crop_to_playable_area=cfg.env.crop_map_to_playable_area,
            raw_crop_to_playable_area=cfg.env.crop_map_to_playable_area,
            action_delays=cfg.env.get('action_delays')
        )

        if self.cfg.evaluate.game_type == 'game_vs_agent':
            self.agent_num = 1
        else:
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
        self._begin_num = cfg.env.begin_num
        self._obs_stat_type = cfg.env.obs_stat_type
        self._pseudo_reward_type = cfg.env.pseudo_reward_type
        self._pseudo_reward_prob = cfg.env.pseudo_reward_prob
        self._ignore_camera = cfg.env.ignore_camera
        assert self._obs_stat_type in ['replay_online', 'self_online', 'replay_last']
        self.reward_helper = RewardHelper(self.agent_num, cfg.env.pseudo_reward_type, cfg.env.pseudo_reward_prob)

        self._reset_flag = False
        # This is the human games statistics used as an input of network
        self._loaded_eval_stats = [None] * self.agent_num
        self.enemy_upgrades = [None] * self.agent_num
        # This is for the statistics of current episode actions and obs
        self._episode_stats = [RealTimeStatistics(self._begin_num) for _ in range(self.agent_num)]

    def load_stat(self, stat, agent_no):
        """
        Set the statistics to be append to every observation of each agent
        stat: stat dict processed by transform_stat
        agent_no: 0 or 1
        """
        self._loaded_eval_stats[agent_no] = GameLoopStatistics(stat, self._begin_num)

    def _merge_stat(self, obs, agent_no, game_loop=None):
        """
        Append the statistics to the observation
        """
        assert self._loaded_eval_stats[agent_no] is not None, "please call load_stat method first"
        if self._obs_stat_type == 'replay_online':
            stat = self._loaded_eval_stats[agent_no].get_input_z_by_game_loop(game_loop=game_loop)
        elif self._obs_stat_type == 'self_online':
            cumulative_stat = self._episode_stats[agent_no].cumulative_statistics
            stat = self._loaded_eval_stats[agent_no].get_input_z_by_game_loop(
                game_loop=None, cumulative_stat=cumulative_stat
            )
        elif self._obs_stat_type == 'replay_last':
            stat = self._loaded_eval_stats[agent_no].get_input_z_by_game_loop(game_loop=None)

        assert set(stat.keys()) == set(['mmr', 'beginning_build_order', 'cumulative_stat'])
        obs['scalar_info'].update(stat)
        return obs

    def _merge_action(self, obs, last_action, agent_no, add_dim=True):
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
        if self.repeat_action_type[agent_no] == last_action_type:
            self.repeat_count[agent_no] += 1
        else:
            self.repeat_action_type[agent_no] = last_action_type
            self.repeat_count[agent_no] = 0
        repeat_tensor = clip_one_hot(torch.LongTensor([self.repeat_count[agent_no]]), 17).squeeze()
        obs['scalar_info']['last_queued'] = torch.cat((obs['scalar_info']['last_queued'], repeat_tensor), dim=0)
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

    def _get_obs(self, obs, agent_no):
        # post process observations returned from sc2env
        last_actions = self.last_actions[agent_no]
        entity_info, entity_raw = self._entity_wrapper.parse(obs)
        new_obs = {
            'scalar_info': self._scalar_wrapper.parse(obs),
            'spatial_info': self._spatial_wrapper.parse(obs),
            'entity_info': entity_info,
            'entity_raw': entity_raw,
            'map_size': [self.map_size[1], self.map_size[0]],  # x,y -> y,x
        }

        def battle_value(obs):
            '''
            The value of destroyed units belong to enemy, sum up minerals and vespene, add for battle baseline
            '''
            kill_value = int(
                np.sum(obs['score_by_category']['killed_minerals']) +
                np.sum(obs['score_by_category']['killed_vespene'])
            )
            return kill_value

        new_obs['battle_value'] = battle_value(obs)

        def score_wrapper(obs):
            '''
            add cumulative_score for baseline
            '''
            score = obs['score_cumulative']
            data = torch.FloatTensor(score)
            return torch.log(data + 1)

        new_obs['score_cumulative'] = score_wrapper(obs)
        new_obs = self._merge_action(new_obs, last_actions)
        new_obs = self._merge_stat(new_obs, agent_no)
        new_obs = get_available_actions_processed_data(new_obs)
        self.enemy_upgrades[agent_no] = get_enemy_upgrades_processed_data(new_obs, self.enemy_upgrades[agent_no])
        new_obs['scalar_info']['enemy_upgrades'] = self.enemy_upgrades[agent_no]
        new_obs = self._mask_obs(new_obs)
        return new_obs

    def _mask_obs(self, obs):
        if self._ignore_camera:
            obs['spatial_info'][1:3] *= 0
            obs['entity_info'][:, 408:410] *= 0
        return obs

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
        # TODO(nyz) deal with step == 0 case for stat and reward
        # temporally set step_mul >= 1
        step_mul = max(1, step_mul)
        if step_mul == 0:
            # repeat last observation and store last action
            # as at least one agent requested this by returning delay=0
            for n in range(self.agent_num):
                if sc2_actions[n]:
                    self._buffered_actions[n].append(sc2_actions[n])
            _, _, obs, rewards, done, info = self._last_output
            for n in range(self.agent_num):
                obs[n] = self._merge_action(obs[n], self.last_actions[n], n, add_dim=False)
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
                    _, r, _, o, info[n] = timestep
                    assert (self.last_actions[n])
                    obs[n] = self._get_obs(o, n)
                    rewards[n] = r
                if due[n]:
                    assert (self.last_actions[n])
                    self._episode_stats[n].update_stat(self.last_actions[n], obs[n], self._episode_steps)
            action_types = [a['action_type'] for a in self.last_actions]
            if self.agent_num == 2:
                battle_values = RewardHelper.BattleValues(
                    self._last_output[2][0]['battle_value'], obs[0]['battle_value'],
                    self._last_output[2][1]['battle_value'], obs[1]['battle_value']
                )
            else:
                battle_values = RewardHelper.BattleValues(0, 0, 0, 0)
            rewards = self.reward_helper.get_pseudo_rewards(
                rewards,
                action_types,
                self._episode_stats,
                self._loaded_eval_stats,
                self._episode_steps,
                battle_values,
                return_list=True
            )
            self._last_output = [self._episode_steps, due, obs, rewards, done, info]
        # as obs may be changed somewhere in parsing
        # we have to return a copy to keep the self._last_ouput intact
        return self._episode_steps, due, copy.deepcopy(obs), rewards, done, info

    def reset(self):
        """
        Reset the env, should be called before start stepping
        Return:
            obs: initial observation
        """
        timesteps = super().reset()
        last_action = {
            'action_type': 0,
            'delay': torch.Tensor([0]),
            'queued': None,
            'selected_units': None,
            'target_units': None,
            'target_location': None
        }
        self.last_actions = [last_action for _ in range(self.agent_num)]
        self.repeat_action_type = [-1] * self.agent_num
        self.repeat_count = [0] * self.agent_num
        obs = []
        for n in range(self.agent_num):
            obs.append(self._get_obs(timesteps[n].observation, n))
        infos = [timestep.game_info for timestep in timesteps]
        env_provided_map_size = infos[0].start_raw.map_size
        env_provided_map_size = [env_provided_map_size.x, env_provided_map_size.y]
        assert tuple(env_provided_map_size) == tuple(self.map_size), \
            "Environment uses a different map size {} compared to config " \
            "{}.".format(env_provided_map_size, self.map_size)

        self._next_obs = [0] * self.agent_num
        # Note: self._episode_steps is updated in SC2Env
        self._episode_steps = 0
        self._reset_flag = True
        self._buffered_actions = [[] for i in range(self.agent_num)]
        self._last_output = [0, [True] * self.agent_num, obs, [0] * self.agent_num, False, infos]
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
