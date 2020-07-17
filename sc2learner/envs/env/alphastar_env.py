import copy
import os
from collections import namedtuple

import numpy as np

import pysc2.env.sc2_env as sc2_env
from pysc2.env.sc2_env import SC2Env
from pysc2.lib.actions import FunctionCall
from pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK
from sc2learner.envs.action.alphastar_action import AlphaStarRawAction
from sc2learner.envs.env.base_env import BaseEnv
from sc2learner.envs.observation.alphastar_obs import ScalarObs, SpatialObs, EntityObs
from sc2learner.envs.other.alphastar_map import get_map_size
from sc2learner.envs.reward.alphastar_reward_runner import AlphaStarRewardRunner
from sc2learner.envs.stat.alphastar_statistics import RealTimeStatistics, GameLoopStatistics
from sc2learner.utils import merge_dicts, read_config

default_config = read_config(os.path.join(os.path.dirname(__file__), '../alphastar_env_default_config.yaml'))
DELAY_INF = 100000


class AlphaStarEnv(BaseEnv, SC2Env):
    timestep = namedtuple('AlphaStarTimestep', ['obs', 'reward', 'done', 'info', 'episode_steps', 'due'])
    info_template = namedtuple('BaseEnvInfo', ['agent_num', 'obs_space', 'act_space', 'rew_space'])

    def __init__(self, cfg: dict) -> None:
        cfg = merge_dicts(default_config.env, cfg)
        self._map_size = get_map_size(cfg.map_name, cropped=cfg.crop_map_to_playable_area)
        cfg.obs_spatial.spatial_resolution = self._map_size
        cfg.action.map_size = self._map_size
        self._cfg = cfg

        self._players, self._agent_num = self._get_players(cfg)

        self._obs_stat_type = cfg.obs_stat_type
        self._ignore_camera = cfg.ignore_camera

        self._obs_scalar = ScalarObs(cfg.obs_scalar)
        self._obs_spatial = SpatialObs(cfg.obs_spatial)
        self._obs_entity = EntityObs(cfg.obs_entity)
        self._begin_num = self._obs_scalar.begin_num
        self._action_helper = AlphaStarRawAction(cfg.action)
        self._reward_helper = AlphaStarRewardRunner(self._agent_num, cfg.pseudo_reward_type, cfg.pseudo_reward_prob)

        self._launch_env_flag = False

    def _get_players(self, cfg):
        if cfg.game_type == 'game_vs_bot':
            agent_num = 1
            players = [
                sc2_env.Agent(sc2_env.Race[cfg.player1.race]),
                sc2_env.Bot(
                    sc2_env.Race[cfg.player2.race], sc2_env.Difficulty[cfg.player2.difficulty],
                    sc2_env.BotBuild[cfg.player2.build]
                )
            ]
        elif cfg.game_type == 'agent_vs_agent':
            agent_num = 2
            players = [sc2_env.Agent(sc2_env.Race[cfg.player1.race]), sc2_env.Agent(sc2_env.Race[cfg.player2.race])]
        else:
            raise KeyError("invalid game_type: {}".format(cfg.game_type))
        return players, agent_num

    def _merge_stat2obs(self, obs, agent_no, game_loop=None):
        assert self._loaded_eval_stat[agent_no] is not None, "please call load_stat method first"
        if self._obs_stat_type == 'replay_online':
            stat = self._loaded_eval_stat[agent_no].get_input_z_by_game_loop(game_loop=game_loop)
        elif self._obs_stat_type == 'self_online':
            cumulative_stat = self._episode_stat[agent_no].cumulative_statistics
            stat = self._loaded_eval_stat[agent_no].get_input_z_by_game_loop(
                game_loop=None, cumulative_stat=cumulative_stat
            )
        elif self._obs_stat_type == 'replay_last':
            stat = self._loaded_eval_stat[agent_no].get_input_z_by_game_loop(game_loop=None)
        else:
            raise ValueError(f"{self._obs_stat_type} unknown!")

        assert set(stat.keys()) == {'mmr', 'beginning_build_order', 'cumulative_stat'}
        obs.update(stat)
        return obs

    def _get_obs(self, obs, agent_no):
        # get last action repeat
        last_action = self._last_action[agent_no]
        last_action = {k: getattr(last_action, k) for k in last_action._fields}
        last_action_type = last_action['action_type']
        if last_action_type == self._repeat_action_type[agent_no]:
            self._repeat_count[agent_no] += 1
        else:
            self._repeat_count[agent_no] = 0
            self._repeat_action_type[agent_no] = last_action_type
        last_action['repeat_count'] = self._repeat_count[agent_no]
        # merge last action
        obs['last_action'] = last_action
        # merge stat
        obs = self._merge_stat2obs(obs, agent_no)
        # transform obs
        entity_info, entity_raw = self._obs_entity._to_agent_processor(obs)
        obs = {
            'scalar_info': self._obs_scalar._to_agent_processor(obs),
            'spatial_info': self._obs_spatial._to_agent_processor(obs),
            'entity_info': entity_info,
            'entity_raw': entity_raw,
            'map_size': [self._map_size[1], self._map_size[0]],  # x,y -> y,x
        }
        obs = self._mask_obs(obs)
        return obs

    def _mask_obs(self, obs):
        if self._ignore_camera:
            obs['spatial_info'][1:3] *= 0
            obs['entity_info'][:, 408:410] *= 0
        return obs

    def _check_action(self, action):
        action_attr = GENERAL_ACTION_INFO_MASK[action.action_type]
        if action_attr['selected_units']:
            if action.selected_units is None or len(action.selected_units) == 0:
                return False
        if action_attr['target_units']:
            if action.target_units is None or len(action.target_units) == 0:
                return False
        return True

    def _get_action(self, action):
        action = copy.deepcopy(action)
        if action is None:
            return FunctionCall.init_with_validation(0, [], raw=True), DELAY_INF, None
        action = self._action_helper._from_agent_processor(action)
        legal = self._check_action(action)
        if not legal:
            # TODO(nyz) more fined solution for illegal action
            print('[WARNING], illegal raw action: {}'.format(action))
            return FunctionCall.init_with_validation(0, [], raw=True), 1, None
        action_type, delay = action[:2]
        args = [v for v in action[2:6] if v is not None]  # queued, selected_units, target_units, target_location
        return FunctionCall.init_with_validation(action_type, args, raw=True), delay, action

    def _get_battle_value(self, raw_obs):
        minerals_ratio = 1.
        vespene_ratio = 1.
        return [
            int(
                np.sum(obs['score_by_category']['killed_minerals']) * minerals_ratio +
                np.sum(obs['score_by_category']['killed_vespene'] * vespene_ratio)
            ) for obs in raw_obs
        ]

    def _launch_env(self) -> None:
        cfg = self._cfg
        agent_interface_format = sc2_env.parse_agent_interface_format(
            feature_screen=cfg.screen_resolution,
            feature_minimap=self._map_size,  # x, y
            crop_to_playable_area=cfg.crop_map_to_playable_area,
            raw_crop_to_playable_area=cfg.crop_map_to_playable_area,
            action_delays=cfg.action_delays
        )

        SC2Env.__init__(
            self,
            map_name=cfg.map_name,
            random_seed=cfg.random_seed,
            step_mul=cfg.default_step_mul,
            players=self._players,
            game_steps_per_episode=cfg.game_steps_per_episode,
            agent_interface_format=agent_interface_format,
            disable_fog=cfg.disable_fog,
            score_index=-1,  # use win/loss reward rather than score
            ensure_available_actions=False,
            realtime=cfg.realtime,
        )

    def reset(self, loaded_stat: list) -> list:
        if not self._launch_env_flag:
            self._launch_env()
        last_action = AlphaStarRawAction.AgentAction(0, 0, None, None, None, None)
        self._reward_helper.reset()
        self._last_action = [last_action for _ in range(self._agent_num)]
        self._episode_stat = [RealTimeStatistics(self._begin_num) for _ in range(self._agent_num)]
        assert len(loaded_stat) == self._agent_num
        self._loaded_eval_stat = [GameLoopStatistics(s, self._begin_num) for s in loaded_stat]
        self._repeat_action_type = [-1] * self._agent_num
        self._repeat_count = [0] * self._agent_num

        timestep = SC2Env.reset(self)
        obs = [self._get_obs(timestep[n].observation, n) for n in range(self._agent_num)]
        self._last_obs = obs
        info = [t.game_info for t in timestep]
        env_provided_map_size = info[0].start_raw.map_size
        env_provided_map_size = [env_provided_map_size.x, env_provided_map_size.y]
        assert tuple(env_provided_map_size) == tuple(self._map_size), \
            "Environment uses a different map size {} compared to config " \
            "{}.".format(env_provided_map_size, self._map_size)
        # Note: self._episode_steps is updated in SC2Env
        self._episode_steps = 0
        self._launch_env_flag = True
        return copy.deepcopy(obs)

    def step(self, action_data: list) -> 'AlphaStarEnv.timestep':
        assert self._launch_env_flag
        # get transformed action and delay
        raw_action, delay, action = list(zip(*[self._get_action(a) for a in action_data]))
        # get step_mul
        step_mul = min(delay)
        assert step_mul >= 0
        # TODO(nyz) deal with step == 0 case for stat and reward
        if step_mul == 0:
            step_mul = 1

        # env step
        timestep = SC2Env.step(self, raw_action, step_mul=step_mul)
        due = [d <= step_mul for d in delay]
        assert any(due), 'at least one of the agents must finish its delay'
        # transform obs, reward and record statistics
        done = False
        obs = [None] * self._agent_num
        reward = [None] * self._agent_num
        info = [None] * self._agent_num
        for n in range(self._agent_num):
            t = timestep[n]
            if t is not None:
                done = done or t.last()  # one of the agents reaches done(last) state
                _, r, _, o, info[n] = t
                obs[n] = self._get_obs(o, n)
                reward[n] = r
                if action[n] is not None and due[n]:
                    self._episode_stat[n].update_stat(action[n], self._last_obs[n], self._episode_steps)
        # Note: pseudo reward must be derived after statistics update
        self.battle_value = self._get_battle_value([t.observation for t in timestep])
        self.action = action
        self.reward = reward
        self.reward = self._reward_helper.get(self)
        # update last state variable
        self._last_obs = obs.copy()
        for n in range(self._agent_num):
            # only valid action update these variable
            if action[n] is not None:
                self._last_action[n] = action[n]

        return AlphaStarEnv.timestep(
            obs=copy.deepcopy(obs),
            reward=self.reward,
            done=done,
            info=info,
            episode_steps=self._episode_steps,
            due=due
        )

    def seed(self, seed: int) -> None:
        """Note: because SC2Env sets up the random seed in input args, we don't implement this method"""
        raise NotImplementedError()

    def info(self) -> 'AlphaStarEnv.info':
        obs_info = {'scalar': self._obs_scalar.info, 'spatial': self._obs_spatial.info, 'entity': self._obs_entity.info}
        info_data = {
            'agent_num': self._agent_num,
            'obs_space': obs_info,
            'act_space': self._action_helper.info,
            'rew_space': self._reward_helper.info,
        }
        return AlphaStarEnv.info_template(**info_data)

    def __repr__(self) -> str:
        obs_str = 'scalar: {}\tspatial: {}\tentity: {}'.format(
            repr(self._obs_scalar), repr(self._obs_spatial), repr(self._obs_entity)
        )
        return 'AlphaStarEnv:\n\
                \tobservation[{}]\n\
                \taction[{}]\n\
                \treward[{}]\n'.format(obs_str, repr(self._action_helper), repr(self._reward_helper))

    def close(self) -> None:
        SC2Env.close(self)

    @property
    def last_action(self) -> list:
        ret = []
        for n in range(self._agent_num):
            handle = self._last_action[n]
            tmp = {}
            for f in handle._fields:
                tmp[f] = getattr(handle, f)
            ret.append(tmp)
        return ret

    @property
    def episode_stat(self) -> RealTimeStatistics:
        return self._episode_stat

    @property
    def episode_steps(self) -> int:
        return self._episode_steps

    @property
    def loaded_eval_stat(self) -> GameLoopStatistics:
        return self._loaded_eval_stat

    @property
    def action(self) -> namedtuple:
        return self._action

    @action.setter
    def action(self, _action: namedtuple) -> None:
        self._action = _action

    @property
    def reward(self) -> list:
        return self._reward

    @reward.setter
    def reward(self, _reward: list) -> None:
        self._reward = _reward

    @property
    def battle_value(self) -> list:
        return self._battle_value

    @battle_value.setter
    def battle_value(self, _battle_value) -> None:
        self._battle_value = _battle_value
