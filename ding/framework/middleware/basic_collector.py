import torch
from typing import TYPE_CHECKING
from ding.envs.env_manager import BaseEnvManager
from ding.torch_utils import to_ndarray, to_tensor
from ding.policy import Policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.worker.buffer import Buffer

if TYPE_CHECKING:
    from ding.framework import Task, Context


def basic_collector(task: "Task", cfg: dict, policy: Policy, env: BaseEnvManager, buffer: Buffer):
    env.seed(cfg.seed)

    epsilon_greedy = None
    if 'eps' in cfg.policy.other:
        eps_cfg = cfg.policy.other.eps
        epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    def _collect(ctx: "Context"):
        if env._closed:
            env.launch()

        ctx.setdefault("collect_env_step", 0)
        ctx.keep("collect_env_step")

        obs = env.ready_obs
        policy_kwargs = {}
        if epsilon_greedy:
            policy_kwargs['eps'] = epsilon_greedy(ctx.collect_env_step)

        policy_output = policy.collect_mode.forward(obs, **policy_kwargs)
        action = to_ndarray({env_id: output['action'] for env_id, output in policy_output.items()})

        timesteps = env.step(action)
        ctx.collect_env_step += len(timesteps)
        timesteps = to_tensor(timesteps, dtype=torch.float32)
        for env_id, timestep in timesteps.items():
            transition = policy.collect_mode.process_transition(obs[env_id], policy_output[env_id], timestep)
            buffer.push(transition)

    return _collect
