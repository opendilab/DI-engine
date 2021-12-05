import sys
from typing import Any, List, Optional, Union

import gym
import gym_soccer
import numpy as np
from ding.envs import BaseEnv, BaseEnvInfo, BaseEnvTimestep
from ding.envs.common.common_function import affine_transform
from ding.envs.common.env_element import EnvElementInfo
from ding.torch_utils import to_list, to_ndarray, to_tensor
from ding.utils import ENV_REGISTRY
from gym.utils import seeding
import copy


@ENV_REGISTRY.register('gym_soccer')
class GymSoccerEnv(BaseEnv):
    default_env_id = ['Soccer-v0', 'SoccerEmptyGoal-v0', 'SoccerAgainstKeeper-v0']

    def __init__(self, cfg: dict = {}) -> None:
        self._cfg = cfg
        self._act_scale = cfg.act_scale
        self._env_id = cfg.env_id
        assert self._env_id in self.default_env_id
        self._init_flag = False
        self._replay_path = './game_log'

    def reset(self) -> np.array:
        if not self._init_flag:
            self._env = gym.make(self._env_id, replay_path=self._replay_path, port=self._cfg.port)  # TODO
            self._init_flag = True
        self._final_eval_reward = 0
        obs = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
        return obs

    def step(self, action: List) -> BaseEnvTimestep:
        if self._act_scale:
            # The continuous action is a Tensor of size = (1,)
            # We indexed at [0] to fetch it as a scalar value
            action[1][0] = affine_transform(action[1][0], min_val=0, max_val=100)
            action[2][0] = affine_transform(action[2][0], min_val=-180, max_val=180)
            action[3][0] = affine_transform(action[3][0], min_val=-180, max_val=180)
            action[4][0] = affine_transform(action[4][0], min_val=0, max_val=100)
            action[5][0] = affine_transform(action[5][0], min_val=-180, max_val=180)

        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        obs = to_ndarray(obs).astype(np.float32)
        # reward wrapped to be transfered to a numpy array with shape (1,)
        rew = to_ndarray([rew])
        # '1' indicates the discrete action is associated with the continuous parameters
        info['action_args_mask'] = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]])
        return BaseEnvTimestep(obs, rew, done, info)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def close(self) -> None:
        self._init_flag = False

    def get_random_action(self):
        # discrete action type: 0, 1, 2
        # continuous action_args:
        #   - power: [0, 100]
        #   - direction: [-180, 180]
        # the action space is (6,), the first indicates discrete action and the remaining indicates continuous action
        # discrete action 0 assotiated with the first and second continuous parameters
        # discrete action 1 assotiated with the third continuous parameter
        # discrete action 2 assotiated with the forth and fifth continuous parameters
        return self._env.action_space.sample()

    def info(self) -> BaseEnvInfo:
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=1,
            obs_space=T(
                (59, ),
                {
                    # [min, max]
                    'min': -1,
                    'max': 1,
                    'dtype': np.float32,
                },
            ),
            act_space=T(
                # the discrete action shape is (3,)
                # however, the continuous action shape is (5,), which is not revealed in the info
                (
                    3,
                ),
                {
                    # [min, max)
                    'min': 0,
                    'max': 3,
                    'dtype': int,
                },
            ),
            rew_space=T(
                (1, ),
                {
                    # [min, max)
                    'min': 0,
                    'max': 2.0,
                    'dtype': int,
                },
            ),
            use_wrappers=None,
        )

    def render(self, close=False):
        self._env.render(close)

    def __repr__(self) -> str:
        return "DI-engine gym soccer Env"

    def replay_log(self, log_path):
        self._env.replay_log(log_path)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './game_log'
        self._replay_path = replay_path

    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        """
        Overview:
            Return a list of all of the environment from input config.
        Arguments:
            - cfg (:obj:`Dict`) Env config, same config where ``self.__init__()`` takes arguments from
        Returns:
            - List of ``cfg`` including all of the collector env's config
        """
        cfg_list = []
        collector_env_num = cfg.pop('collector_env_num')
        port_pool = list(range(6000, 9999))
        port_candidates = np.random.choice(port_pool, size=collector_env_num, replace=False)
        for i in range(collector_env_num):
            cfg_copy = copy.deepcopy(cfg)
            cfg_copy.port = port_candidates[i]
            cfg_list.append(cfg_copy)
        return cfg_list

    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        """
        Overview:
            Return a list of all of the environment from input config.
        Arguments:
            - cfg (:obj:`Dict`) Env config, same config where ``self.__init__()`` takes arguments from
        Returns:
            - List of ``cfg`` including all of the evaluator env's config
        """
        cfg_list = []
        evaluator_env_num = cfg.pop('evaluator_env_num')
        port_pool = list(range(6000, 9999))
        port_candidates = np.random.choice(port_pool, size=evaluator_env_num, replace=False)
        for i in range(evaluator_env_num):
            cfg_copy = copy.deepcopy(cfg)
            cfg_copy.port = port_candidates[i]
            cfg_list.append(cfg_copy)
        return cfg_list
