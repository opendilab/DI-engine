from typing import Callable, Dict, TYPE_CHECKING
import copy
import gym
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from easydict import EasyDict
from collections import namedtuple
if TYPE_CHECKING:
    from jax._src.prng import PRNGKeyArray


class Agent:

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    @staticmethod
    def apply_train(loss_fn: Callable, model, optimizer, data: Dict, log_optim: bool = False) -> Dict:
        """
            Inplace modify param.
        """
        grad, info = jax.grad(loss_fn, has_aux=True)(model.param, data)
        # TODO(nyz) jax.pmean for allreduce
        model.param, optim_info = optimizer.step(grad, model.param, log_optim)
        if log_optim:
            info.update(optim_info)  # such as grad norm
        return info

    def random_transition(
            self, rng_key: "PRNGKeyArray", obs_space: gym.spaces.Space, action_space: gym.spaces.Space
    ) -> Dict:
        obs = obs_space.sample().clip(-1, 1)
        action = action_space.sample()
        if np.isscalar(action):
            action = np.array([action])
        next_obs = obs_space.sample().clip(-1, 1)
        reward = np.random.random()
        done = np.random.random() >= 0.5
        info = {}
        timestep = (next_obs, reward, done, info)
        return self.process_transition(obs, action, timestep)


class NNState:

    def __init__(self, key, dummy_data, create_model_fn):
        self.model_fn = hk.transform(lambda x: create_model_fn()(x))
        self.model_fn = hk.without_apply_rng(self.model_fn)
        self.param = self.model_fn.init(key, dummy_data)

    def forward(self, x):
        return self.model_fn.apply(self.param, x)

    def learn(self, param, x):
        return self.model_fn.apply(param, x)
