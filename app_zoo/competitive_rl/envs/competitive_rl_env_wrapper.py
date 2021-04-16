import cv2
import gym
import os.path as osp
import numpy as np
from typing import Union, Optional
from collections import deque
from competitive_rl.pong.builtin_policies import get_builtin_agent_names, single_obs_space, single_act_space, get_random_policy, get_rule_based_policy
from competitive_rl.utils.policy_serving import Policy
from nervex.envs import ObsTransposeWrapper, WarpFrame, FrameStack

def get_compute_action_function_ours(agent_name, num_envs=1):
    resource_dir = osp.join(osp.dirname(__file__), "resources", "pong")
    if agent_name == "STRONG":
        return Policy(
            single_obs_space, single_act_space, num_envs,
            osp.join(resource_dir, "checkpoint-strong.pkl"),
            use_light_model=False
        )
    if agent_name == "MEDIUM":
        return Policy(
            single_obs_space, single_act_space, num_envs,
            osp.join(resource_dir, "checkpoint-medium.pkl"),
            use_light_model=True
        )
    if agent_name == "ALPHA_PONG":
        return Policy(
            single_obs_space, single_act_space, num_envs,
            osp.join(resource_dir, "checkpoint-alphapong.pkl"),
            use_light_model=False
        )
    if agent_name == "WEAK":
        return Policy(
            single_obs_space, single_act_space, num_envs,
            osp.join(resource_dir, "checkpoint-weak.pkl"),
            use_light_model=True
        )
    if agent_name == "RANDOM":
        return get_random_policy(num_envs)
    if agent_name == "RULE_BASED":
        return get_rule_based_policy(num_envs)
    raise ValueError("Unknown agent name: {}".format(agent_name))


class BuiltinOpponentWrapper(gym.Wrapper):
    def __init__(self, env: 'gym.Env', num_envs: int = 1) -> None:  # noqa
        super().__init__(env)
        self.agents = {agent_name: get_compute_action_function_ours(
            agent_name, num_envs) for agent_name in get_builtin_agent_names()}
        self.agent_names = list(self.agents)
        self.prev_opponent_obs = None
        self.current_opponent_name = "RULE_BASED"
        self.current_opponent = self.agents[self.current_opponent_name]
        self.observation_space = env.observation_space[0]
        self.action_space = env.action_space[0]
        self.num_envs = num_envs

    def reset_opponent(self, agent_name: str) -> None:
        assert agent_name in self.agent_names, (agent_name, self.agent_names)
        self.current_opponent_name = agent_name
        self.current_opponent = self.agents[self.current_opponent_name]

    def step(self, action):
        tuple_action = (
            action.item(),
            self.current_opponent(self.prev_opponent_obs)
        )
        obs, rew, done, info = self.env.step(tuple_action)
        self.prev_opponent_obs = obs[1]
        # if done.ndim == 2:
        #     done = done[:, 0]
        # return obs[0], rew[:, 0].reshape(-1, 1), done.reshape(-1, 1), info
        return obs[0], rew[0], done, info

    def reset(self):
        obs = self.env.reset()
        self.prev_opponent_obs = obs[1]
        return obs[0]

    def seed(self, s):
        self.env.seed(s)


# def wrap_env(env_id, builtin_wrap, opponent, frame_stack=4, warp_frame=True):
def wrap_env(env_id, builtin_wrap, opponent, frame_stack=0, warp_frame=False):
    """Configure environment for DeepMind-style Atari. The observation is
    channel-first: (c, h, w) instead of (h, w, c).

    :param str env_id: the atari environment id.
    :param bool episode_life: wrap the episode life wrapper.
    :param bool clip_rewards: wrap the reward clipping wrapper.
    :param int frame_stack: wrap the frame stacking wrapper.
    :param bool scale: wrap the scaling observation wrapper.
    :param bool warp_frame: wrap the grayscale + resize observation wrapper.
    :return: the wrapped atari environment.
    """
    env = gym.make(env_id)
    if builtin_wrap:
        env = BuiltinOpponentWrapper(env)
        env.reset_opponent(opponent)
    env = ObsTransposeWrapper(env)

    if warp_frame:
        env = WarpFrame(env)
    if frame_stack:
        env = FrameStack(env, frame_stack)
    return env
