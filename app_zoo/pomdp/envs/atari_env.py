from typing import Any, List, Union, Sequence
import copy
import torch
import gym
import numpy as np
from nervex.envs import BaseEnv, register_env, BaseEnvTimestep, BaseEnvInfo
from nervex.envs.common.env_element import EnvElement, EnvElementInfo
from nervex.torch_utils import to_tensor, to_ndarray, to_list
from .atari_wrappers import wrap_deepmind
import pdb

def PomdpEnv(cfg):
    '''
    For debug purpose, create an env follow openai gym standard so it can be widely test by
    other library with same environment setting in nerveX
    env = PomdpEnv(cfg)
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    '''
    env = wrap_deepmind(
        cfg.env_id,
        frame_stack=cfg.frame_stack,
        episode_life=cfg.is_train,
        clip_rewards=cfg.is_train,
        warp_frame=cfg.warp_frame,
        use_ram=cfg.use_ram,
        render=cfg.render,
        pomdp=cfg.pomdp,
    )
    return env


class PomdpAtariEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self.env_num = 4
        # self._env = wrap_deepmind(
        #     cfg.env_id,
        #     frame_stack=cfg.frame_stack,
        #     # episode_life=cfg.is_train,
        #     episode_life=True,
        #     clip_rewards=cfg.is_train,
        #     warp_frame=cfg.warp_frame,
        #     use_ram=cfg.use_ram,
        #     render=cfg.render,
        #     pomdp=cfg.pomdp,
        #     reward_scale=cfg.reward_scale
        # )

        self._env = [wrap_deepmind(
            cfg.env_id,
            frame_stack=cfg.frame_stack,
            # episode_life=cfg.is_train,
            episode_life=True,
            clip_rewards=cfg.is_train,
            warp_frame=cfg.warp_frame,
            use_ram=cfg.use_ram,
            render=cfg.render,
            pomdp=cfg.pomdp,
            reward_scale=cfg.reward_scale
        ) for _ in range(self.env_num)]

    def reset(self) -> Sequence:
        if hasattr(self, '_seed'):
            np.random.seed(self._seed)
            _ = [_env.seed(self._seed + i) for i, _env in enumerate(self._env)]
        c_obs = [_env.reset() for _env in self._env]
        obs = np.concatenate(c_obs)
        self._final_eval_reward = 0.
        return obs

    def close(self) -> None:
        self._env.close()

    def seed(self, seed: int) -> None:
        self._seed = seed

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        c_obs = []
        c_rew = []
        _done = False
        for _env,_action in zip(self._env, action):
            obs, rew, done, info = _env.step(_action)
            c_rew.append(rew)
            c_obs.append(obs)
            _done = _done or done

        self._final_eval_reward += sum(c_rew)
        obs = np.concatenate(c_obs)
        rew = to_ndarray([sum(c_rew)])  # wrapped to be transfered to a Tensor with shape (1,)
        if _done:
            info['final_eval_reward'] = self._final_eval_reward

        return BaseEnvTimestep(obs, rew, _done, info)

    def info(self) -> BaseEnvInfo:
        rew_range = self._env[0].reward_range
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=1,
            obs_space=T(self._env[0].observation_space.shape, {'dtype': np.float32}, None, None),
            act_space=T(self.env_num * (self._env[0].action_space.n, ), {'dtype': np.float32}, None, None),
            rew_space=T(1, {
                'min': rew_range[0],
                'max': rew_range[1],
                'dtype': np.float32
            }, None, None),
        )

    def __repr__(self) -> str:
        return "nerveX POMDP Atari Env({})".format(self._cfg.env_id)

    @staticmethod
    def create_actor_env_cfg(cfg: dict) -> List[dict]:
        actor_env_num = cfg.pop('actor_env_num', 1)
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(actor_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num', 1)
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]


register_env('pomdp', PomdpAtariEnv)
