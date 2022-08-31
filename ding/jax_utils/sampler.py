import numpy as np
import jax.numpy as jnp


class EpsGreedySampler:

    def __call__(self, val, eps: float):
        if np.random.random() > eps:
            return val.argmax(axis=-1)
        else:
            N = val.shape[-1]
            return jnp.array(np.random.randint(0, N, size=val.shape[:-1]))


class ArgmaxSampler:

    def __call__(self, val):
        return val.argmax(axis=-1)
