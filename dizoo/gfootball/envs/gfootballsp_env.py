import copy
from collections import namedtuple
from typing import Any, List, Union

import gfootball
import gfootball.env as football_env
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from dizoo.gfootball.envs.obs.encoder import FeatureEncoder
from dizoo.gfootball.envs.obs.gfootball_obs import FullObs
from dizoo.gfootball.envs.action.gfootball_action import GfootballSpAction


@ENV_REGISTRY.register('gfootball_sp')
class GfootballEnv(BaseEnv):

    timestep = namedtuple('GfootballTimestep', ['obs', 'reward', 'done', 'info'])
    info_template = namedtuple('GFootballEnvInfo', ['obs_space', 'act_space', 'rew_space'])

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self.save_replay = self._cfg.save_replay
        # self.env_name = cfg.get("env_name", "11_vs_11_kaggle")
        self.gui = self._cfg.render
        self._obs_helper = FullObs(cfg)
        self._action_helper = GfootballSpAction(cfg)
        self._launch_env_flag = False
        self._encoder = FeatureEncoder()
        self.is_evaluator = self._cfg.get("is_evaluator", False)
        if self.is_evaluator:
            self.env_name = "11_vs_11_hard_stochastic"
            self.right_role_num = 0
        else:
            self.env_name = "11_vs_11_kaggle"
            self.right_role_num = 1

    def _make_env(self):
        self._env = football_env.create_environment(
            env_name=self.env_name,
            representation='raw',
            stacked=False,
            logdir='/tmp/football',
            write_goal_dumps=False,
            write_full_episode_dumps=self.save_replay,
            write_video=self.save_replay,
            render=self.gui,
            number_of_right_players_agent_controls=self.right_role_num
        )
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        self._launch_env_flag = True
        if self.is_evaluator:
            self._final_eval_reward = [0, 0]
        else:
            self._final_eval_reward = [0, 0]

    def reset(self) -> np.ndarray:
        if not self._launch_env_flag:
            self._make_env()
            self._init_flag = True
        self._env.reset()
        obs = self._env.observation()
        if self.is_evaluator:
            self._prev_obs = obs[0]
            obs = self._encoder.encode(obs[0])
            return [obs, obs]
        else:
            self._prev_obs, self.prev_obs_opponent = obs
            obs_ = self._encoder.encode(obs[0])
            obs_opponent = self._encoder.encode(obs[1])
            return [obs_, obs_opponent]

    def close(self) -> None:
        if self._launch_env_flag:
            self._env.close()
        self._launch_env_flag = False

    def seed(self, seed: int, dynamic_seed: int = None) -> None:
        self._seed = seed
        if dynamic_seed:
            self._dynamic_seed = dynamic_seed

    def step(self, action) -> 'GfootballEnv.timestep':
        action = to_ndarray(action)
        # action = self.process_action(action)  # process
        raw_obs, raw_rew, done, info = self._env.step(action)
        if self.is_evaluator:
            raw_obs = raw_obs[0]
            rew = GfootballEnv.calc_reward(raw_rew, self._prev_obs, raw_obs)
            obs = to_ndarray(self._encoder.encode(raw_obs))
            rew = [rew, rew]
            obs = [obs, obs]
            self._final_eval_reward[0] += raw_rew
            self._final_eval_reward[1] += raw_rew
        else:
            rew = GfootballEnv.calc_reward(raw_rew[0], self._prev_obs, raw_obs[0])
            rew_oppo = GfootballEnv.calc_reward(raw_rew[1], self._prev_obs, raw_obs[1])
            rew = [rew, rew_oppo]
            obs = [to_ndarray(self._encoder.encode(raw_obs[0])), to_ndarray(self._encoder.encode(raw_obs[1]))]
            self._final_eval_reward[0] += raw_rew[0]
            self._final_eval_reward[1] += raw_rew[1]

        if done:
            if self.is_evaluator:
                info['final_eval_reward'] = self._final_eval_reward
            else:
                info[0]['final_eval_reward'] = self._final_eval_reward[0]
                info[1]['final_eval_reward'] = self._final_eval_reward[1]

        return BaseEnvTimestep(obs, rew, done, info)

    def info(self) -> BaseEnvInfo:
        info_data = {
            'obs_space': self._obs_helper.info,
            'act_space': self._action_helper.info,
            'rew_space': EnvElementInfo(
                shape=1,
                value={
                    'min': np.float64("-inf"),
                    'max': np.float64("inf"),
                    'dtype': np.float32
                },
            ),
        }
        return GfootballEnv.info_template(**info_data)

    def __repr__(self) -> str:
        return "DI-engine Gfootball Env({})".format(self.env_name)

    @staticmethod
    def calc_reward(rew, prev_obs, obs):
        """
        Reward disign referred to [football-pairs](https://github.com/seungeunrho/football-paris/blob/main/rewarders/rewarder_basic.py)
        """
        ball_x, ball_y, ball_z = obs['ball']
        MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
        PENALTY_Y, END_Y = 0.27, 0.42

        ball_position_r = 0.0
        if (-END_X <= ball_x and ball_x < -PENALTY_X) and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
            ball_position_r = -2.0
        elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (-END_Y < ball_y and ball_y < END_Y):
            ball_position_r = -1.0
        elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (-END_Y < ball_y and ball_y < END_Y):
            ball_position_r = 0.0
        elif (PENALTY_X < ball_x and ball_x <= END_X) and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
            ball_position_r = 2.0
        elif (MIDDLE_X < ball_x and ball_x <= END_X) and (-END_Y < ball_y and ball_y < END_Y):
            ball_position_r = 1.0
        else:
            ball_position_r = 0.0

        left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(prev_obs["left_team_yellow_card"])
        right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(prev_obs["right_team_yellow_card"])
        yellow_r = right_yellow - left_yellow

        win_reward = 0.0
        if obs['steps_left'] == 0:
            [my_score, opponent_score] = obs['score']
            if my_score > opponent_score:
                win_reward = 1.0

        reward = 5.0 * win_reward + 5.0 * rew + 0.003 * ball_position_r + yellow_r

        return reward

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_cfg = copy.deepcopy(cfg)
        collector_env_num = collector_cfg.pop('collector_env_num', 1)
        collector_cfg.is_evaluator = False
        return [collector_cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_cfg = copy.deepcopy(cfg)
        evaluator_env_num = evaluator_cfg.pop('evaluator_env_num', 1)
        evaluator_cfg.is_evaluator = True
        return [evaluator_cfg for _ in range(evaluator_env_num)]
