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


class FrameStack(gym.Wrapper):
    """Stack n_frames last frames.

    :param gym.Env env: the environment to wrap.
    :param int n_frames: the number of frames to stack.
    """

    def __init__(self, env, n_frames):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.tuple.Tuple):
            obs_space = (obs_space,)
        shape = (n_frames, ) + obs_space[0].shape
        self.observation_space = gym.spaces.tuple.Tuple([gym.spaces.Box(
            low=np.min(obs_space[0].low),
            high=np.max(obs_space[0].high),
            shape=shape,
            dtype=obs_space[0].dtype
        ) for _ in range(len(obs_space))])
        if len(self.observation_space) == 1:
            self.observation_space = self.observation_space[0]

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        # the original wrapper use `LazyFrames` but since we use np buffer,
        # it has no effect
        return np.stack(self.frames, axis=0)


class WarpFrame(gym.ObservationWrapper):
    """Warp frames to 84x84 as done in the Nature paper and later work.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        self.size = 84
        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.tuple.Tuple):
            obs_space = (obs_space,)
        self.observation_space = gym.spaces.tuple.Tuple([gym.spaces.Box(
            low=np.min(obs_space[0].low),
            high=np.max(obs_space[0].high),
            shape=(self.size, self.size),
            dtype=obs_space[0].dtype
        ) for _ in range(len(obs_space))])
        if len(self.observation_space) == 1:
            self.observation_space = self.observation_space[0]

    def observation(self, frame):
        """returns the current observation from a frame"""
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)


class ObsTransposeWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        if isinstance(obs_space, gym.spaces.tuple.Tuple):
            self.observation_space = gym.spaces.Box(
                low=np.min(obs_space[0].low),
                high=np.max(obs_space[0].high),
                shape=(len(obs_space), obs_space[0].shape[2], obs_space[0].shape[0], obs_space[0].shape[1]),
                dtype=obs_space[0].dtype
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=np.min(obs_space.low),
                high=np.max(obs_space.high),
                shape=(obs_space.shape[2], obs_space.shape[0], obs_space.shape[1]),
                dtype=obs_space.dtype
            )

    def observation(self, obs: Union[tuple, np.ndarray]):
        if isinstance(obs, tuple):
            new_obs = []
            for i in range(len(obs)):
                new_obs.append(obs[i].transpose(2, 0, 1))
            obs = np.stack(new_obs)
        else:
            obs = obs.transpose(2, 0, 1)
        return obs


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
