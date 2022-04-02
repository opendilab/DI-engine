from typing import TYPE_CHECKING, Callable, List
from easydict import EasyDict
from functools import reduce

from ding.policy import Policy, get_random_policy
from ding.envs import BaseEnvManager
from ding.framework import task
from .functional import inferencer, rolloutor

if TYPE_CHECKING:
    from ding.framework import Context


class StepCollector:

    def __init__(self, cfg: EasyDict, policy, env: BaseEnvManager, random_collect_size: int = 0) -> None:
        self.cfg = cfg
        self.env = env
        self.policy = policy
        self.random_collect_size = random_collect_size
        self._transitions = [[] for _ in range(self.env.env_num)]
        self._inferencer = task.wrap(inferencer(cfg, policy, env))
        self._rolloutor = task.wrap(rolloutor(cfg, policy, env, self._transitions))

    def _reset_transitions(self):
        for item in self._transitions:
            item.clear()

    def __call__(self, ctx: "Context") -> None:
        old = ctx.env_step
        if self.random_collect_size > 0 and old < self.random_collect_size:
            target_size = self.random_collect_size - old
            random_policy = get_random_policy(self.cfg, self.policy, self.env)
            current_inferencer = task.wrap(inferencer(self.cfg, random_policy, self.env))
        else:
            # compatible with old config, a train sample = unroll_len step
            target_size = self.cfg.policy.collect.n_sample * self.cfg.policy.collect.unroll_len
            current_inferencer = self._inferencer

        while True:
            current_inferencer(ctx)
            self._rolloutor(ctx)
            if ctx.env_step - old >= target_size:
                ctx.trajectories = sum(self._transitions, [])
                lengths = [len(t) for t in self._transitions]
                ctx.trajectory_end_idx = [reduce(lambda x, y: x + y, lengths[:i + 1]) for i in range(len(lengths))]
                ctx.trajectory_end_idx = [t - 1 for t in ctx.trajectory_end_idx]
                self._reset_transitions()
                break
