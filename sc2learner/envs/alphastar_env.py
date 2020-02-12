import torch
import pysc2.envs.sc2_env as sc2_env
from pysc2.envs.sc2_env import SC2Env
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from sc2learner.envs.observations.alphastar_obs_wrapper import SpatialObsWrapper, ScalarObsWrapper, EntityObsWrapper,\
    transform_spatial_data, transform_scalar_data, transform_entity_data


class AlphastarEnv(SC2Env):

    def __init__(self, cfg):
        agent_interface_format = sc2_env.parse_agent_interface_format(
                feature_screen=cfg.env.resolution, feature_minimap=cfg.env.resolution)
        players = [
            sc2_env.Agent(sc2_env.Race[cfg.env.home_race]),
            sc2_env.Bot(sc2_env.Race[cfg.env.away_race]),
        ]
        super(AlphastarEnv, self).__init__(
            map_name=cfg.env.map_name,
            random_seed=cfg.env.random_seed,
            step_mul=cfg.env.default_step_mul,
            players=players,
            game_step_per_episode=cfg.game_step_per_episode,
            agent_interface_format=agent_interface_format,
            disable_fog=cfg.env.disable_fog,
            ensure_available_actions=False,
        )
        self.spatial_wrapper = SpatialObsWrapper(transform_spatial_data())
        self.entity_wrapper = EntityObsWrapper(transform_entity_data())
        template_obs, template_replay, template_act = transform_scalar_data()
        self.scalar_wrapper = ScalarObsWrapper(template_obs)
        self.template_act = template_act

        self.use_stat = cfg.common.use_stat
        if self.use_stat:
            self.stat = self._init_stat(cfg.env.stat_path, cfg.env.beginning_build_order_num)

    def _init_stat(path, begin_num):
        stat = torch.load(path)
        if stat['beginning_build_order'].shape[0] < begin_num:
            B, N = stat['beginning_build_order'].shape
            B0 = begin_num - B
            stat['beginning_build_order'] = torch.cat([stat['beginning_build_order'], torch.zeros(B0, N)])
        return stat

    def _merge_stat(self, obs):
        obs['mmr'] = self.stat['mmr']
        obs['beginning_build_order'] = self.stat['beginning_build_order']
        obs['cumulative_stat'] = self.stat['cumulative_stat']
        return obs

    def _merge_action(self, obs, last_action):
        N = obs['entity_info'].shape[0]
        if obs['entity_info'] is None:
            obs['entity_info'] = torch.cat([obs['entity_info'], torch.zeros(N, 4)], dim=1)
            return obs
        last_action_type = last_action['action_type']
        last_delay = last_action['delay']
        last_queued = last_action['queued']
        last_queued = last_queued if isinstance(last_queued, torch.Tensor) else torch.LongTensor([2])  # 2 as 'none'
        obs['scalar_info']['last_delay'] = self.template_act[0]['op'](torch.LongTensor(last_delay)).squeeze()
        obs['scalar_info']['last_queued'] = self.template_act[1]['op'](torch.LongTensor(last_queued)).squeeze()
        obs['scalar_info']['last_action_type'] = self.template_act[2]['op'](
            torch.LongTensor(last_action_type)).squeeze()

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
        entity_info, entity_raw = self.entity_wrapper.parse(obs)
        new_obs = {
            'scalar_info': self.scalar_wrapper.parse(obs),
            'spatial_info': self.spatial_wrapper.parse(obs),
            'entity_info': entity_info,
            'entity_raw': entity_raw,
        }
        new_obs = self._merge_action(new_obs, last_actions)
        if self.use_stat:
            new_obs = self._merge_stat(new_obs)
        return new_obs

    def _get_action(self, actions):
        action_type = actions['action_type']
        delay = actions['delay']
        #return FunctionCall(action_type, args, raw=True), delay
        return FUNCTIONS.no_op(), delay

    def step(self, actions):
        last_actions = actions
        transformed_action, delay = self._get_action(actions)
        obs, reward, done, info = super.step(transformed_action, step_mul=delay)
        obs = self._get_obs(obs, last_actions)
        return obs, reward, done, info

    def reset(self):
        obs = super.reset()
        obs = self._get_obs(obs)
        return obs
