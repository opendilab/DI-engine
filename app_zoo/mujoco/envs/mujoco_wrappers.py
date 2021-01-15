import gym
import numpy as np


class RunningMeanStd(object):
    # Refer to https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    def __init__(self, epsilon=1e-4, shape=()):
        self._epsilon = epsilon
        self._shape = shape
        self.reset()

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        new_count = batch_count + self._count
        mean_delta = batch_mean - self._mean
        new_mean = self._mean + mean_delta * batch_count / new_count
        # this method for calculating new variable might be numerically unstable
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(mean_delta) * self._count * batch_count / new_count
        new_var = m2 / new_count
        self._mean = new_mean
        self._var = new_var
        self._count = new_count
    
    def reset(self):
        self._mean = np.zeros(self._shape, 'float64')
        self._var = np.ones(self._shape, 'float64')
        self._count = self._epsilon

    @property
    def mean(self) -> np.ndarray:
        return self._mean
    
    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self._var) + self._epsilon


class ObsNormEnv(gym.ObservationWrapper):
    """Normalize observations according to running mean and std.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        self.data_count = 0
        self.clip_range = (-3, 3)
        self.rms = RunningMeanStd(shape=env.observation_space.shape)

    def step(self, action):
        self.data_count += 1
        observation, reward, done, info = self.env.step(action)
        self.rms.update(observation)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        if self.data_count > 30:
            return np.clip((observation - self.rms.mean) / self.rms.std, self.clip_range[0], self.clip_range[1])
        else:
            return observation
    
    def reset(self, **kwargs):
        self.data_count = 0
        self.rms.reset()
        observation = self.env.reset(**kwargs)
        return self.observation(observation)


class RewardNormEnv(gym.RewardWrapper):
    """Normalize reward according to running std.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env, reward_discount):
        super().__init__(env)
        self.cum_reward = np.zeros((1,), 'float64')
        self.reward_discount = reward_discount
        self.data_count = 0
        self.rms = RunningMeanStd(shape=(1,))

    def step(self, action):
        self.data_count += 1
        observation, reward, done, info = self.env.step(action)
        reward = np.array([reward], 'float64')
        self.cum_reward = self.cum_reward * self.reward_discount + reward
        self.rms.update(self.cum_reward)
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        if self.data_count > 30:
            return float(reward / self.rms.std)
        else:
            return float(reward)
    
    def reset(self, **kwargs):
        self.cum_reward = 0.
        self.data_count = 0
        self.rms.reset()
        return self.env.reset(**kwargs)


def wrap_mujoco(env_id, norm_obs=True, norm_reward=True) -> gym.Env:
    r"""
    Overview:
        Wrap Mujoco Env to preprocess env step's return info, e.g. observation normalization, reward normalization, etc.
    Arguments:
        - env_id (:obj:`str`): Mujoco environment id, for example "HalfCheetah-v3"
        - norm_obs (:obj:`EasyDict`): Whether to normalize observation or not
        - norm_reward (:obj:`EasyDict`): Whether to normalize reward or not. For evaluator, environment's reward \
            should not be normalized: Either ``norm_reward`` is None or ``norm_reward.use_norm`` is False can do this.
    Returns:
        - wrapped_env (:obj:`gym.Env`): The wrapped mujoco environment
    """
    env = gym.make(env_id)
    if norm_obs is not None and norm_obs.use_norm:
        env = ObsNormEnv(env)
    if norm_reward is not None and norm_reward.use_norm:
        env = RewardNormEnv(env, norm_reward.reward_discount)
    return env
