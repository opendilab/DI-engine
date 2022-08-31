from collections import namedtuple
import jax
import jax.numpy as jnp

q_1step_td_data = namedtuple('q_1step_td_data', ['q', 'next_q', 'act', 'next_act', 'reward', 'done'])


def q_1step_td_error(data, gamma):
    q, next_q, act, next_act, reward, done = data
    done = done.astype(jnp.float32)
    batch_range = jnp.arange(q.shape[0])

    value = q[batch_range, act]
    target_value = next_q[batch_range, next_act]
    return_ = reward + gamma * target_value * (1 - done)
    td_error_per_sample = (value - return_) ** 2
    loss = jnp.mean(td_error_per_sample)
    return loss, td_error_per_sample
