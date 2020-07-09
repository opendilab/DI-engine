from collections import namedtuple
import copy
import numpy as np
import torch

import pysc2.env.sc2_env as sc2_env
from pysc2.env.sc2_env import SC2Env
from pysc2.lib.actions import FunctionCall
from sc2learner.envs.env.base_env import BaseEnv
from sc2learner.envs.observation.alphastar_obs import ScalarObs, SpatialObs, EntityObs
from sc2learner.envs.action.alphastar_action import AlphaStarRawAction
from sc2learner.envs.reward.alphastar_reward import AlphaStarReward
from sc2learner.envs.other.alphastar_map import get_map_size
from sc2learner.envs.stat.alphastar_statistics import RealTimeStatistics, GameLoopStatistics


class AlphaStarEnv(BaseEnv, SC2Env):
    timestep = namedtuple('AlphaStarTimestep', ['obs', 'reward', 'done', 'info', 'episode_steps', 'due'])
    info = namedtuple('BaseEnvInfo', ['agents_num', 'obs_space', 'act_space', 'rew_space'])

    def __init__(self, cfg: dict) -> None:
        self.map_size = get_map_size(cfg.map_name, cropped=cfg.crop_map_to_playable_area)
        cfg.obs_spatial.spatial_resolution = self.map_size
        cfg.action.map_size = self.map_size
        self.cfg = cfg

        agent_interface_format = sc2_env.parse_agent_interface_format(
            feature_screen=cfg.screen_resolution,
            feature_minimap=self.map_size,  # x, y
            crop_to_playable_area=cfg.crop_map_to_playable_area,
            raw_crop_to_playable_area=cfg.crop_map_to_playable_area,
            action_delays=cfg.action_delays
        )

        players, self.agent_num = self._get_players(cfg)

        SC2Env.__init__(
            self,
            map_name=cfg.map_name,
            random_seed=cfg.random_seed,
            step_mul=cfg.default_step_mul,
            players=players,
            game_steps_per_episode=cfg.game_steps_per_episode,
            agent_interface_format=agent_interface_format,
            disable_fog=cfg.disable_fog,
            score_index=-1,  # use win/loss reward rather than score
            ensure_available_actions=False,
            realtime=cfg.realtime,
        )
        self._obs_stat_type = cfg.obs_stat_type
        self._ignore_camera = cfg.ignore_camera

        self._obs_scalar = ScalarObs(cfg.obs_scalar)
        self._obs_spatial = SpatialObs(cfg.obs_spatial)
        self._obs_entity = EntityObs(cfg.obs_entity)
        self._begin_num = self._obs_scalar.begin_num
        self._action_helper = AlphaStarRawAction(cfg.action)
        self._reward_helper = AlphaStarReward(self.agent_num, cfg.pseudo_reward_type, cfg.pseudo_reward_prob)

        self._reset_flag = False

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

        assert set(stat.keys()) == set(['mmr', 'beginning_build_order', 'cumulative_stat'])
        obs.update(stat)
        return obs

    def _get_obs(self, obs, agent_no):
        # get last action repeat
        last_action = self._last_action[agent_no]
        last_action_type = last_action['action_type'].item()
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
            'map_size': [self.map_size[1], self.map_size[0]],  # x,y -> y,x
        }
        obs = self._mask_obs(obs)
        return obs

    def _mask_obs(self, obs):
        if self._ignore_camera:
            obs['spatial_info'][1:3] *= 0
            obs['entity_info'][:, 408:410] *= 0
        return obs

    def _get_action(self, action):
        action = copy.deepcopy(action)
        action = self._action_helper._from_agent_processor(action)
        action_type, delay = action[:2]
        args = [v for v in action[2:6] if v is not None]  # queued, selected_units, target_units, target_location
        return FunctionCall.init_with_validation(action_type, args, raw=True), delay

    def _get_battle_value(self, raw_obs):
        minerals_ratio = 1.
        vespene_ratio = 1.

        def battle_fn(obs):
            return int(
                np.sum(obs['score_by_category']['killed_minerals']) * minerals_ratio +
                np.sum(obs['score_by_category']['killed_vespene'] * vespene_ratio)
            )

        return [battle_fn(raw_obs[n]) for n in range(self.agent_num)]

    def _get_pseudo_rewards(self, reward, battle_value, action):
        action_type = [a['action_type'] for a in action]
        if self.agent_num == 1:
            battle_value = AlphaStarReward.BattleValues(
                self._last_battle_value[0], battle_value[0], self._last_battle_value[1], battle_value[1]
            )
        else:
            battle_value = AlphaStarReward.BattleValues(0, 0, 0, 0)
        return self._reward_helper._to_agent_processor(
            reward,
            action_type,
            self._episode_stat,
            self._loaded_eval_stat,
            self._episode_steps,
            battle_value,
            return_list=True
        )

    def reset(self, loaded_stat: list) -> dict:
        last_action = {
            'action_type': torch.LongTensor([0]),
            'delay': torch.LongTensor([0]),
            'queued': None,
            'selected_units': None,
            'target_units': None,
            'target_location': None
        }
        self._last_action = [last_action for _ in range(self.agent_num)]
        self._last_battle_value = [0] * self.agent_num
        self._episode_stat = [RealTimeStatistics(self._begin_num) for _ in range(self.agent_num)]
        assert len(loaded_stat) == self.agent_num
        self._loaded_eval_stat = [GameLoopStatistics(s, self._begin_num) for s in loaded_stat]
        self._repeat_action_type = [-1] * self.agent_num
        self._repeat_count = [0] * self.agent_num

        timestep = SC2Env.reset(self)
        obs = [self._get_obs(timestep[n].observation, n) for n in range(self.agent_num)]
        self._last_obs = obs
        info = [t.game_info for t in timestep]
        env_provided_map_size = info[0].start_raw.map_size
        env_provided_map_size = [env_provided_map_size.x, env_provided_map_size.y]
        assert tuple(env_provided_map_size) == tuple(self.map_size), \
            "Environment uses a different map size {} compared to config " \
            "{}.".format(env_provided_map_size, self.map_size)
        # Note: self._episode_steps is updated in SC2Env
        self._episode_steps = 0
        self._reset_flag = True
        return copy.deepcopy(obs)

    def step(self, action_data: list) -> 'AlphaStarEnv.timestep':
        assert self._reset_flag
        # get transformed action and delay
        raw_action, delay = list(zip(*[self._get_action(a) for a in action_data]))
        action = [t['action'] for t in action_data]
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
        obs = [None] * self.agent_num
        reward = [None] * self.agent_num
        info = [None] * self.agent_num
        for n in range(self.agent_num):
            t = timestep[n]
            if t is not None:
                done = done or t.last()  # one of the agents reaches done(last) state
                _, r, _, o, info[n] = t
                obs[n] = self._get_obs(o, n)
                reward[n] = r
            if due[n]:
                self._episode_stat[n].update_stat(action[n], self._last_obs[n], self._episode_steps)
        # Note: pseudo reward must be derived after statistics update
        battle_value = self._get_battle_value([t.observation for t in timestep])
        reward = self._get_pseudo_rewards(reward, battle_value, action)
        # update state variable
        self._last_action = action
        self._last_obs = obs
        self._last_battle_value = battle_value

        return AlphaStarEnv.timestep(
            obs=copy.deepcopy(obs), reward=reward, done=done, info=info, episode_steps=self._episode_steps, due=due
        )

    def seed(self, seed: int) -> None:
        raise NotImplementedError

    def info(self) -> 'AlphaStarEnv.info':
        pass

    def __repr__(self) -> str:
        pass

    def close(self) -> None:
        SC2Env.close(self)
