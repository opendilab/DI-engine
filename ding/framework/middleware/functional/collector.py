from typing import TYPE_CHECKING, Callable, List
from easydict import EasyDict
import torch
import treetensor.torch as ttorch
from ding.envs import BaseEnvManager
from ding.policy import Policy
from ding.framework import task

if TYPE_CHECKING:
    from ding.framework import Context


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


def rolloutor(cfg: EasyDict, policy: Policy, env: BaseEnvManager, transitions: List[List[ttorch.Tensor]]) -> Callable:

    def _rollout(ctx):
        timesteps = env.step(ctx.action)
        ctx.env_step += len(timesteps)
        timesteps = [t.tensor() for t in timesteps]
        # TODO abnormal env step
        for i, timestep in enumerate(timesteps):
            transition = policy.process_transition(ctx.obs[i], ctx.inference_output[i], timestep)
            transition = ttorch.as_tensor(transition)  # TBD
            transition.collect_train_iter = ttorch.as_tensor([ctx.train_iter])
            transitions[timestep.env_id].append(transition)
            print(transition)
            print(timestep.done)
            if timestep.done:
                policy.reset([timestep.env_id])
                ctx.env_episode += 1
        # TODO log

    return _rollout


def episode_collector(cfg: EasyDict, policy: Policy, env: BaseEnvManager) -> Callable:
    _inferencer = inferencer(cfg, policy, env)
    _rolloutor = rolloutor(cfg, policy, env)

    def _collect(ctx: "Context"):
        old = ctx.env_episode
        while True:
            _inferencer(ctx)
            _rolloutor(ctx)
            if ctx.env_episode - old > cfg.policy.collect.n_episode:
                break

    return _collect


# TODO battle collector
