import cv2
import gym
import os.path as osp
import numpy as np
from typing import Union, Optional
from collections import deque
from competitive_rl.pong.builtin_policies import get_builtin_agent_names, single_obs_space, single_act_space, get_random_policy, get_rule_based_policy
from competitive_rl.utils.policy_serving import Policy


def get_compute_action_function_ours(agent_name, num_envs=1):
    resource_dir = osp.join(osp.dirname(__file__), "resources", "pong")
    if agent_name == "STRONG":
        return Policy(
            single_obs_space,
            single_act_space,
            num_envs,
            osp.join(resource_dir, "checkpoint-strong.pkl"),
            use_light_model=False
        )
    if agent_name == "MEDIUM":
        return Policy(
            single_obs_space,
            single_act_space,
            num_envs,
            osp.join(resource_dir, "checkpoint-medium.pkl"),
            use_light_model=True
        )
    if agent_name == "ALPHA_PONG":
        return Policy(
            single_obs_space,
            single_act_space,
            num_envs,
            osp.join(resource_dir, "checkpoint-alphapong.pkl"),
            use_light_model=False
        )
    if agent_name == "WEAK":
        return Policy(
            single_obs_space,
            single_act_space,
            num_envs,
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
        self.agents = {
            agent_name: get_compute_action_function_ours(agent_name, num_envs)
            for agent_name in get_builtin_agent_names()
        }
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
        tuple_action = (action.item(), self.current_opponent(self.prev_opponent_obs))
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


def wrap_env(env_id, builtin_wrap, opponent, frame_stack=4, warp_frame=True, only_info=False):
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
    if not only_info:
        env = gym.make(env_id)
        if builtin_wrap:
            env = BuiltinOpponentWrapper(env)
            env.reset_opponent(opponent)

        if warp_frame:
            env = WarpFrameWrapperCompetitveRl(env, builtin_wrap)
        if frame_stack:
            env = FrameStackWrapperCompetitiveRl(env, frame_stack, builtin_wrap)
        return env
    else:
        wrapper_info = ''
        if builtin_wrap:
            wrapper_info += BuiltinOpponentWrapper.__name__ + '\n'
        if warp_frame:
            wrapper_info = WarpFrameWrapperCompetitveRl.__name__ + '\n'
        if frame_stack:
            wrapper_info = FrameStackWrapperCompetitiveRl.__name__ + '\n'
        return wrapper_info


class WarpFrameWrapperCompetitveRl(gym.ObservationWrapper):
    """Warp frames to 84x84 as done in the Nature paper and later work.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env, builtin_wrap):
        super().__init__(env)
        self.size = 84
        obs_space = env.observation_space
        self.builtin_wrap = builtin_wrap
        if builtin_wrap:
            # single player
            self.observation_space = gym.spaces.Box(
                low=np.min(obs_space.low),
                high=np.max(obs_space.high),
                shape=(self.size, self.size),
                dtype=obs_space.dtype
            )
        else:
            # double player
            self.observation_space = gym.spaces.tuple.Tuple(
                [
                    gym.spaces.Box(
                        low=np.min(obs_space[0].low),
                        high=np.max(obs_space[0].high),
                        shape=(self.size, self.size),
                        dtype=obs_space[0].dtype
                    ) for _ in range(len(obs_space))
                ]
            )

    def observation(self, frame):
        """returns the current observation from a frame"""
        if self.builtin_wrap:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            return cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
        else:
            frames = []
            for one_frame in frame:
                one_frame = cv2.cvtColor(one_frame, cv2.COLOR_RGB2GRAY)
                one_frame = cv2.resize(one_frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
                frames.append(one_frame)
            return frames


class FrameStackWrapperCompetitiveRl(gym.Wrapper):
    """Stack n_frames last frames.

    :param gym.Env env: the environment to wrap.
    :param int n_frames: the number of frames to stack.
    """

    def __init__(self, env, n_frames, builtin_wrap):
        super().__init__(env)
        self.n_frames = n_frames

        self.builtin_wrap = builtin_wrap
        obs_space = env.observation_space
        if self.builtin_wrap:
            self.frames = deque([], maxlen=n_frames)
            shape = (n_frames, ) + obs_space.shape
            self.observation_space = gym.spaces.Box(
                low=np.min(obs_space.low), high=np.max(obs_space.high), shape=shape, dtype=obs_space.dtype
            )
        else:
            self.frames = [deque([], maxlen=n_frames) for _ in range(len(obs_space))]
            shape = (n_frames, ) + obs_space[0].shape
            self.observation_space = gym.spaces.tuple.Tuple(
                [
                    gym.spaces.Box(
                        low=np.min(obs_space[0].low),
                        high=np.max(obs_space[0].high),
                        shape=shape,
                        dtype=obs_space[0].dtype
                    ) for _ in range(len(obs_space))
                ]
            )

    def reset(self):
        if self.builtin_wrap:
            obs = self.env.reset()
            for _ in range(self.n_frames):
                self.frames.append(obs)
            return self._get_ob(self.frames)
        else:
            obs = self.env.reset()
            for i, one_obs in enumerate(obs):
                for _ in range(self.n_frames):
                    self.frames[i].append(one_obs)
            return np.stack([self._get_ob(self.frames[i]) for i in range(len(obs))])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.builtin_wrap:
            self.frames.append(obs)
            return self._get_ob(self.frames), reward, done, info
        else:
            for i, one_obs in enumerate(obs):
                self.frames[i].append(one_obs)
            return np.stack([self._get_ob(self.frames[i]) for i in range(len(obs))], axis=0), reward, done, info

    @staticmethod
    def _get_ob(frames):
        # the original wrapper use `LazyFrames` but since we use np buffer,
        # it has no effect
        return np.stack(frames, axis=0)
