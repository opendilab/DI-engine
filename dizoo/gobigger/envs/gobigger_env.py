from typing import Any, List, Union, Optional, Tuple
import time
import copy
import math
import cv2
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.torch_utils import to_tensor, to_ndarray, to_list
from ding.utils import ENV_REGISTRY
from gobigger.server import Server
from gobigger.render import EnvRender


def one_hot_np(value: int, num_cls: int):
    ret = np.zeros(num_cls)
    ret[value] = 1
    return ret


@ENV_REGISTRY.register('gobigger')
class GoBiggerEnv(BaseEnv):
    config = dict(
        player_num_per_team=2,
        team_num=3,
        match_time=1200,
        map_height=1000,
        map_width=1000,
        resize_height=160,
        resize_width=160,
        spatial=True,
        train=True,
    )

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._player_num_per_team = cfg.player_num_per_team
        self._team_num = cfg.team_num
        self._player_num = self._player_num_per_team * self._team_num
        self._match_time = cfg.match_time
        self._map_height = cfg.map_height
        self._map_width = cfg.map_width
        self._resize_height = cfg.resize_height
        self._resize_width = cfg.resize_width
        self._spatial = cfg.spatial
        self._train = cfg.train
        self._last_team_size = None
        self._init_flag = False

    def _launch_game(self) -> Server:
        server = Server(self._cfg)
        server.start()
        render = EnvRender(server.map_width, server.map_height, use_spatial=self._spatial)
        server.set_render(render)
        self._player_names = sum(server.get_player_names_with_team(), [])
        return server

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = self._launch_game()
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            # self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            pass
            # self._env.seed(self._seed)
        self._final_eval_reward = [0. for _ in range(self._team_num)]
        self._env.reset()
        raw_obs = self._env.obs()
        obs = self._obs_transform(raw_obs)
        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def render(self) -> None:
        if not self._init_flag:
            self._env = self._launch_game()
            self._init_flag = True

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def _obs_transform(self, obs: tuple) -> list:
        global_state, player_state = obs
        # global
        # 'border': [map_width, map_height] fixed map size
        total_time_feat = one_hot_np(round(min(1200, global_state['total_time']) / 100), 13)
        last_time_feat = one_hot_np(round(min(1200, global_state['last_time']) / 100), 13)
        # only use leaderboard rank
        leaderboard_feat = np.zeros((self._team_num, self._team_num))
        for idx, (team_name, team_size) in enumerate(global_state['leaderboard'].items()):
            team_name_number = int(team_name[-1])
            leaderboard_feat[idx, team_name_number] = 1
        leaderboard_feat = leaderboard_feat.reshape(-1)
        global_feat = np.concatenate([total_time_feat, last_time_feat, leaderboard_feat])
        # player
        obs = []
        for n, value in player_state.items():
            if self._spatial:
                player_spatial_feat = []
                for c, item in enumerate(value['feature_layers']):
                    # cv2.imwrite('before_{}_{}.jpg'.format(n, c), item*255)
                    one_channel_item = item[..., np.newaxis].astype(np.float32)
                    resize_item = cv2.resize(one_channel_item, (self._resize_width, self._resize_height))
                    player_spatial_feat.append(resize_item)
                    # cv2.imwrite('after_{}_{}.jpg'.format(n, c), resize_item.astype(np.uint8)*255)
                player_spatial_feat = np.stack(player_spatial_feat, axis=-1).transpose(2, 0, 1)

            team_name_feat = one_hot_np(int(value['team_name'][-1]), self._team_num)
            ori_left_top_x, ori_left_top_y, ori_right_bottom_x, ori_right_bottom_y = value['rectangle']
            left_top_x, right_bottom_x = ori_left_top_x / self._map_width, ori_right_bottom_x / self._map_width
            left_top_y, right_bottom_y = ori_left_top_y / self._map_height, ori_right_bottom_y / self._map_height
            rectangle_feat = np.stack([left_top_x, left_top_y, right_bottom_x, right_bottom_y])
            player_scalar_feat = np.concatenate([rectangle_feat, team_name_feat])

            player_unit_feat = []
            unit_type_mapping = {'food': 0, 'thorn': 1, 'spore': 2, 'clone': 3}
            raw_overlap = {}
            for unit_type in value['overlap']:
                raw_overlap_one_type = list(value['overlap'][unit_type])
                if raw_overlap_one_type is None:
                    raw_overlap_one_type = []
                raw_overlap[unit_type] = copy.deepcopy(raw_overlap_one_type)
                for unit in raw_overlap_one_type:
                    if unit_type == 'clone':
                        position, radius = unit['position'], unit['radius']
                        player_name, team_name = unit['player'], unit['team']
                        player_number, team_nubmer = int(player_name[-1]), int(team_name[-1])
                    else:
                        position, radius = unit['position'], unit['radius']
                        player_number, team_nubmer = self._player_num, self._team_num  # placeholder
                    radius_feat = one_hot_np(round(min(10, math.sqrt(radius))), 11)
                    position = [
                        (position[0] - ori_left_top_x) / (ori_right_bottom_x - ori_left_top_x),
                        (position[1] - ori_right_bottom_y) / (ori_left_top_y - ori_right_bottom_y)
                    ]
                    position_feat = np.stack(position)
                    player_feat = one_hot_np(player_number, self._player_num + 1)
                    team_feat = one_hot_np(team_nubmer, self._team_num + 1)
                    player_unit_feat_item = np.concatenate([position_feat, radius_feat, player_feat, team_feat])
                    player_unit_feat.append(player_unit_feat_item)
            if len(player_unit_feat) <= 200:
                padding_num = 200 - len(player_unit_feat)
                padding_player_unit_feat = np.zeros((padding_num, player_unit_feat[0].shape[0]))
                player_unit_feat = np.stack(player_unit_feat)
                player_unit_feat = np.concatenate([player_unit_feat, padding_player_unit_feat])
            else:
                player_unit_feat = np.stack(player_unit_feat)[-200:]

            obs.append(
                {
                    'scalar_obs': np.concatenate([global_feat, player_scalar_feat]).astype(np.float32),
                    'unit_obs': player_unit_feat.astype(np.float32),
                    'unit_num': len(player_unit_feat),
                    'collate_ignore_raw_obs': copy.deepcopy({'overlap': raw_overlap}),
                }
            )
            if self._spatial:
                obs[-1]['spatial_obs'] = player_spatial_feat.astype(np.float32)
        team_obs = []
        for i in range(self._team_num):
            team_obs.append(obs[i * self._player_num_per_team:(i + 1) * self._player_num_per_team])
        return team_obs

    def _act_transform(self, act: list) -> dict:
        act = [item.tolist() for item in act]
        act = sum(act, [])
        # the element of act can be int scalar or structed object
        return {n: self._to_raw_action(a) if np.isscalar(a) else a for n, a in zip(self._player_names, act)}

    @staticmethod
    def _to_raw_action(act: int) -> Tuple[float, float, int]:
        assert 0 <= act < 16
        # -1, 0, 1, 2(noop, eject, split, gather)
        # 0, 1, 2, 3(up, down, left, right)
        action_type, direction = act // 4, act % 4
        action_type = action_type - 1
        if direction == 0:
            x, y = 0, 1
        elif direction == 1:
            x, y = 0, -1
        elif direction == 2:
            x, y = -1, 0
        elif direction == 3:
            x, y = 1, 0
        return [x, y, action_type]

    def _get_reward(self, obs: tuple) -> list:
        global_state, _ = obs
        if self._last_team_size is None:
            team_reward = [np.array([0.]) for __ in range(self._team_num)]
        else:
            reward = []
            for n in self._player_names:
                team_name = str(int(n) // self._player_num_per_team)
                last_size = self._last_team_size[team_name]
                cur_size = global_state['leaderboard'][team_name]
                reward.append(np.array([cur_size - last_size]))
            team_reward = []
            for i in range(self._team_num):
                team_reward_item = sum(reward[i * self._player_num_per_team:(i + 1) * self._player_num_per_team])
                if self._train:
                    team_reward_item = np.clip(team_reward_item / 2, -1, 1)
                team_reward.append(team_reward_item)
        self._last_team_size = global_state['leaderboard']
        return team_reward

    def step(self, action: list) -> BaseEnvTimestep:
        action = self._act_transform(action)
        done = self._env.step(action)
        raw_obs = self._env.obs()
        obs = self._obs_transform(raw_obs)
        rew = self._get_reward(raw_obs)
        info = [{} for _ in range(self._team_num)]

        for i, team_reward in enumerate(rew):
            self._final_eval_reward[i] += team_reward
        if done:
            for i in range(self._team_num):
                info[i]['final_eval_reward'] = self._final_eval_reward[i]
        return BaseEnvTimestep(obs, rew, done, info)

    def info(self) -> BaseEnvInfo:
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=self._player_num,
            obs_space=T(
                {
                    'spatial': (self._player_num + 3, self._resize_width, self._resize_height),
                    'scalar': (42, ),
                    'unit': (188, 21),  # unit is dynamic list
                },
                {
                    'min': 0,
                    'max': 1,
                    'dtype': np.float32,
                },
            ),
            # [min, max)
            act_space=T(
                (1, ),
                {
                    'min': 0,
                    'max': 16,
                    'dtype': int,
                },
            ),
            rew_space=T(
                (1, ),
                {
                    'min': -1000.0,
                    'max': 1000.0,
                    'dtype': np.float32,
                },
            ),
            use_wrappers=None,
        )

    def __repr__(self) -> str:
        return "DI-engine GoBigger Env"

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        raise NotImplementedError
