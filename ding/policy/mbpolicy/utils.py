from typing import Tuple, Union

import torch
from torch import Tensor, Size

def flatten_batch(x: Tensor, nonbatch_dims=1) -> Tuple[Tensor, Size]:
    # (b1,b2,..., X) => (B, X)
    if nonbatch_dims > 0:
        batch_dim = x.shape[:-nonbatch_dims]
        x = torch.reshape(x, (-1,) + x.shape[-nonbatch_dims:])
        return x, batch_dim
    else:
        batch_dim = x.shape
        x = torch.reshape(x, (-1,))
        return x, batch_dim

def unflatten_batch(x: Tensor, batch_dim: Union[Size, Tuple]) -> Tensor:
    # (B, X) => (b1,b2,..., X)
    x = torch.reshape(x, batch_dim + x.shape[1:])
    return x

def q_evaluation(obss, actions, q_critic_fn):
    obss, dim = flatten_batch(obss, 1)
    actions, _ = flatten_batch(actions, 1)
    q_values = q_critic_fn(obss, actions)
    # twin critic
    if isinstance(q_values, list):
        return [
            unflatten_batch(q_values[0], dim), 
            unflatten_batch(q_values[1], dim)
        ] 
    return unflatten_batch(q_values, dim)

def rollout(obs, actor_fn, horizon, env):
    obss        = [obs]
    actions     = []
    rewards     = []
    aug_rewards = []    # -temperature*logprob
    dones       = [torch.zeros_like(obs.sum(-1))]
    for _ in range(horizon):
        action, aug_reward = actor_fn(obs)
        # done: probability of termination
        reward, obs, done = env(obs, action)
        reward = reward + aug_reward
        obss.append(obs)
        actions.append(action)
        rewards.append(reward)
        aug_rewards.append(aug_reward)
        dones.append(done)
    action, aug_reward = actor_fn(obs)
    actions.append(action)
    aug_rewards.append(aug_reward)
    return (
        torch.stack(obss), 
        torch.stack(actions), 
        # rewards is an empty list when horizon=0
        torch.stack(rewards) if rewards else torch.tensor(rewards), 
        torch.stack(aug_rewards), 
        torch.stack(dones)
    )
