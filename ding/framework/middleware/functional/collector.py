from typing import TYPE_CHECKING, Callable
from easydict import EasyDict
import torch
import treetensor.torch as ttorch
from ding.envs import BaseEnvManager
from ding.policy import Policy
from ding.data import Buffer
from ding.framework import task

if TYPE_CHECKING:
    from ding.framework import Context

from ding.rl_utils import get_epsilon_greedy_fn


def eps_greedy_handler(cfg: EasyDict) -> Callable:
    eps_cfg = cfg.policy.other.eps
    handle = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    def _eps_greedy(ctx: "Context"):
        ctx.collect_kwargs['eps'] = handle(ctx.env_step)
        yield
        try:
            ctx.collect_kwargs.pop('eps')
        except:  # noqa
            pass

    return _eps_greedy


def inferencer(cfg: EasyDict, policy: Policy, env: BaseEnvManager) -> Callable:
    env.seed(cfg.seed)

    def _inference(ctx: "Context"):
        if env.closed:
            env.launch()

        obs = ttorch.as_tensor(env.ready_obs).to(dtype=ttorch.float32)
        ctx.obs = obs
        # TODO mask necessary rollout

        obs = {i: obs[i] for i in range(obs.shape[0])}  # TBD
        inference_output = policy.forward(obs, **ctx.collect_kwargs)
        ctx.action = [v['action'].numpy() for v in inference_output.values()]  # TBD
        ctx.inference_output = inference_output

    return _inference


def rolloutor(cfg: EasyDict, policy: Policy, env: BaseEnvManager, buffer_: Buffer) -> Callable:

    def _rollout(ctx):
        timesteps = env.step(ctx.action)
        ctx.env_step += len(timesteps)
        timesteps = [t.tensor(dtype=torch.float32) for t in timesteps]
        # TODO abnormal env step
        for i, timestep in enumerate(timesteps):
            transition = policy.process_transition(ctx.obs[i], ctx.inference_output[i], timestep)
            transition = ttorch.as_tensor(transition)  # TBD
            transition.collect_train_iter = ttorch.as_tensor([ctx.train_iter])
            buffer_.push(transition)
            if timestep.done:
                policy.reset([timestep.env_id])
                ctx.env_episode += 1
        # TODO log

    return _rollout


def step_collector(cfg: EasyDict, policy: Policy, env: BaseEnvManager, buffer_: Buffer) -> Callable:
    _inferencer = inferencer(cfg, policy, env)
    _rolloutor = rolloutor(cfg, policy, env, buffer_)

    def _collect(ctx: "Context"):
        old = ctx.env_step
        while True:
            _inferencer(ctx)
            _rolloutor(ctx)
            if ctx.env_step - old > cfg.policy.collect.n_sample * cfg.policy.collect.unroll_len:
                break

    return _collect


def episode_collector(cfg: EasyDict, policy: Policy, env: BaseEnvManager, buffer_: Buffer) -> Callable:
    _inferencer = inferencer(cfg, policy, env)
    _rolloutor = rolloutor(cfg, policy, env, buffer_)

    def _collect(ctx: "Context"):
        old = ctx.env_episode
        while True:
            _inferencer(ctx)
            _rolloutor(ctx)
            if ctx.env_episode - old > cfg.policy.collect.n_episode:
                break

    return _collect


# TODO battle collector
