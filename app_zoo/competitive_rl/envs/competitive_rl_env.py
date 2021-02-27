from typing import Any, Union, List
import copy
import torch
import numpy as np
import gym
import competitive_rl

from nervex.envs import BaseEnv, register_env, BaseEnvTimestep, BaseEnvInfo
from nervex.envs.common.env_element import EnvElement, EnvElementInfo
from nervex.envs.common.common_function import affine_transform
from nervex.torch_utils import to_tensor, to_ndarray, to_list
from .competitive_rl_env_wrapper import BuiltinOpponentWrapper, wrap_env

competitive_rl.register_competitive_envs()

"""
The observation spaces:
cPong-v0: Box(210, 160, 3)
cPongDouble-v0: Tuple(Box(210, 160, 3), Box(210, 160, 3))
cCarRacing-v0: Box(96, 96, 1)
cCarRacingDouble-v0: Box(96, 96, 1)

The action spaces:
cPong-v0: Discrete(3)
cPongDouble-v0: Tuple(Discrete(3), Discrete(3))
cCarRacing-v0: Box(2,)
cCarRacingDouble-v0: Dict(0:Box(2,), 1:Box(2,))

cPongTournament-v0
"""

class CompetitiveRlEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        # self._use_act_scale = cfg.use_act_scale
        self._env_id = self._cfg.env_id

        # opponent_type is used to control builtin opponent agent, which is useful in evaluator.
        is_evaluator = self._cfg.get("is_evaluator", False)
        opponent_type = None
        if is_evaluator:
            opponent_type = self._cfg.get("opponent_type", None)
        self._builtin_wrap = self._env_id == "cPongDouble-v0" and is_evaluator and opponent_type == "builtin"
        opponent = self._cfg.get('eval_opponent', 'RULE_BASED')

        self._env = wrap_env(self._env_id, self._builtin_wrap, opponent)  # issue

        # if self._env_id == "cPongDouble-v0" and is_evaluator and opponent_type == "builtin":
        #     self._env = BuiltinOpponentWrapper(self._env)
        #     opponent = self._cfg.get('eval_opponent', 'RULE_BASED')
        #     self._env.reset_opponent(opponent)

    def reset(self) -> np.ndarray:
        if hasattr(self, '_seed'):
            self._env.seed(self._seed)
        obs = self._env.reset()

        obs = to_ndarray(obs)
        # print("!! obs(reset)", type(obs))

        assert not isinstance(obs, list)
        obs = self.process_obs(obs)  # process

        self._final_eval_reward = 0.
        return obs

    def close(self) -> None:
        self._env.close()

    def seed(self, seed: int) -> None:
        self._seed = seed

    def step(self, action: Union[torch.Tensor, np.ndarray, list]) -> BaseEnvTimestep:
        action = to_ndarray(action)
        # if self._use_act_scale:
        #     action_range = self.info().act_space.value
        #     action = affine_transform(action, min_val=action_range['min'], max_val=action_range['max'])
        
        action = self.process_action(action)  # process

        obs, rew, done, info = self._env.step(action)

        if self._env_id == "cPongDouble-v0" and not self._builtin_wrap:
            rew = rew[0]  # Left player's reward  todo car race double env?

        self._final_eval_reward += rew
        
        obs = to_ndarray(obs)
        # print("!! obs(step)", type(obs))

        assert not isinstance(obs, list)
        obs = self.process_obs(obs)  # process

        rew = to_ndarray([rew])  # Wrapped to be transformed to an ndarray with shape (1,)
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        return BaseEnvTimestep(obs, rew, done, info)

    def info(self) -> BaseEnvInfo:
        reward_range = self._env.reward_range
        observation_space = self._env.observation_space
        action_space = self._env.action_space
        agent_num = len(observation_space) if isinstance(observation_space, gym.spaces.tuple.Tuple) else 1
        if agent_num > 1:
            obs_shape = (len(observation_space), ) + observation_space[0].shape
            act_shape = (len(action_space), ) + action_space[0].shape
        else:
            obs_shape = observation_space.shape
            act_shape = action_space.shape
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=agent_num,
            obs_space=T(
                obs_shape, {'dtype': np.float32}, None, None
            ),
            act_space=T(
                act_shape, {'dtype': np.float32}, None, None
            ),
            rew_space=T(1, {
                'min': reward_range[0],
                'max': reward_range[1],
                'dtype': np.float32
            }, None, None),
        )

    def __repr__(self) -> str:
        return "nerveX Competitve RL Env({})".format(self._cfg.env_id)

    @staticmethod
    def create_actor_env_cfg(cfg: dict) -> List[dict]:
        actor_cfg = copy.deepcopy(cfg)
        actor_env_num = actor_cfg.pop('actor_env_num', 1)
        actor_cfg.is_evaluator = False
        return [actor_cfg for _ in range(actor_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_cfg = copy.deepcopy(cfg)
        evaluator_env_num = evaluator_cfg.pop('evaluator_env_num', 1)
        evaluator_cfg.is_evaluator = True
        return [evaluator_cfg for _ in range(evaluator_env_num)]
    
    def process_action(self, action: np.ndarray) -> Union[tuple, dict, np.ndarray]:
        # If in double agent env, transfrom action passed in from outside to tuple or dict type.
        if self._env_id == "cPongDouble-v0" and not self._builtin_wrap:
            return (action[0].squeeze(), action[1].squeeze())
        elif self._env_id == "cCarRacingDouble-v0":
            return {0: action[0].squeeze(), 1: action[1].squeeze()}
        else:
            return action.squeeze()
    
    def process_obs(self, obs: Union[tuple, np.ndarray]) -> Union[tuple, np.ndarray]:
        # Copy observation for car racing double agent env, in case to be in alignment with pong double agent env.
        if self._env_id == "cCarRacingDouble-v0":
            obs = np.stack([obs, copy.deepcopy(obs)])
        # if isinstance(obs, tuple):
        #     obs = np.stack(obs)
        return obs

register_env('competitive_rl', CompetitiveRlEnv)
