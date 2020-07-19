import copy
import os
from collections import namedtuple

import numpy as np

import pysc2.env.sc2_env as sc2_env
from pysc2.env.sc2_env import SC2Env
from sc2learner.envs.env.base_env import BaseEnv
from sc2learner.envs.other.alphastar_map import get_map_size
from sc2learner.envs.action.alphastar_action_runner import AlphaStarRawActionRunner
from sc2learner.envs.reward.alphastar_reward_runner import AlphaStarRewardRunner
from sc2learner.envs.observation.alphastar_obs_runner import AlphaStarObsRunner
from sc2learner.envs.stat.alphastar_statistics import RealTimeStatistics, GameLoopStatistics
from sc2learner.utils import merge_dicts, read_config

default_config = read_config(os.path.join(os.path.dirname(__file__), '../alphastar_env_default_config.yaml'))


class AlphaStarEnv(BaseEnv, SC2Env):
    timestep = namedtuple('AlphaStarTimestep', ['obs', 'reward', 'done', 'info', 'episode_steps', 'due'])
    info_template = namedtuple('BaseEnvInfo', ['agent_num', 'obs_space', 'act_space', 'rew_space'])

    def __init__(self, cfg: dict) -> None:
        cfg = merge_dicts(default_config.env, cfg)
        self._map_size = get_map_size(cfg.map_name, cropped=cfg.crop_map_to_playable_area)
        cfg.obs_spatial.spatial_resolution = self._map_size
        cfg.action.map_size = self._map_size
        self._players, self._agent_num = self._get_players(cfg)
        cfg.agent_num = self._agent_num
        self._cfg = cfg

        self._obs_helper = AlphaStarObsRunner(cfg)
        self._begin_num = self._obs_helper._obs_scalar.begin_num
        self._action_helper = AlphaStarRawActionRunner(cfg)
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
        self._reward_helper.reset()
        self._obs_helper.reset()
        self._action_helper.reset()
        self._episode_stat = [RealTimeStatistics(self._begin_num) for _ in range(self._agent_num)]
        assert len(loaded_stat) == self._agent_num
        self._loaded_eval_stat = [GameLoopStatistics(s, self._begin_num) for s in loaded_stat]

        timestep = SC2Env.reset(self)
        self._raw_obs = [timestep[n].observation for n in range(self._agent_num)]
        obs = self._obs_helper.get(self)
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
        self.agent_action = action_data
        raw_action, delay, action = self._action_helper.get(self)
        # get step_mul
        step_mul = min(delay)
        assert step_mul >= 0
        # TODO(nyz) deal with step == 0 case for stat and reward
        if step_mul == 0:
            step_mul = 1
        due = [d <= step_mul for d in delay]
        assert any(due), 'at least one of the agents must finish its delay'
        # Note: record statistics must be executed before env step
        for n in range(self._agent_num):
            if action[n] is not None and due[n]:
                self._episode_stat[n].update_stat(action[n], self._last_obs[n], self._episode_steps)

        # env step
        timestep = SC2Env.step(self, raw_action, step_mul=step_mul)

        # transform obs, reward and record statistics
        self.raw_obs = [timestep[n].observation for n in range(self._agent_num)]
        obs = self._obs_helper.get(self)
        self.reward = [timestep[n].reward for n in range(self._agent_num)]
        info = [timestep[n].game_info for n in range(self._agent_num)]
        done = any([timestep[n].last() for n in range(self._agent_num)])
        # Note: pseudo reward must be derived after statistics update
        self.action = action
        self.reward = self._reward_helper.get(self)
        # update last state variable
        self._last_obs = obs

        return AlphaStarEnv.timestep(
            obs=copy.deepcopy(obs),
            reward=self.reward,
            done=done,
            info=info,
            episode_steps=self.episode_steps,
            due=due
        )

    def seed(self, seed: int) -> None:
        """Note: because SC2Env sets up the random seed in input args, we don't implement this method"""
        raise NotImplementedError()

    def info(self) -> 'AlphaStarEnv.info':
        info_data = {
            'agent_num': self._agent_num,
            'obs_space': self._obs_helper.info,
            'act_space': self._action_helper.info,
            'rew_space': self._reward_helper.info,
        }
        return AlphaStarEnv.info_template(**info_data)

    def __repr__(self) -> str:
        return 'AlphaStarEnv:\n\
                \tobservation[{}]\n\
                \taction[{}]\n\
                \treward[{}]\n'.format(repr(self._obs_helper), repr(self._action_helper), repr(self._reward_helper))

    def close(self) -> None:
        SC2Env.close(self)

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
    def raw_obs(self) -> list:
        return self._raw_obs

    @raw_obs.setter
    def raw_obs(self, _raw_obs) -> None:
        self._raw_obs = _raw_obs

    @property
    def agent_action(self) -> list:
        return self._agent_action

    @agent_action.setter
    def agent_action(self) -> None:
        return self._agent_action
