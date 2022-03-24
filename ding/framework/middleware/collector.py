from typing import TYPE_CHECKING, Callable, List
from easydict import EasyDict
from functools import reduce

from ding.policy import Policy
from ding.envs import BaseEnvManager
from ding.framework import task
from .functional import inferencer, rolloutor

if TYPE_CHECKING:
    from ding.framework import Context


class StepCollector:

    def __init__(self, cfg: EasyDict, policy: Policy, env: BaseEnvManager) -> None:
        self.cfg = cfg
        self.env = env
        self._transitions = [[] for _ in range(self.env.env_num)]
        self._inferencer = inferencer(cfg, policy, env)
        self._rolloutor = rolloutor(cfg, policy, env, self._transitions)

    def _reset_transitions(self):
        for item in self._transitions:
            item.clear()

    def __call__(self, ctx: "Context") -> None:
        old = ctx.env_step
        while True:
            self._inferencer(ctx)
            self._rolloutor(ctx)
            if ctx.env_step - old > self.cfg.policy.collect.n_sample * self.cfg.policy.collect.unroll_len:
                ctx.trajectories = sum(self._transitions, [])
                lengths = [len(t) for t in self._transitions]
                ctx.trajectory_end_idx = [reduce(lambda x, y: x + y, lengths[:i + 1]) for i in range(len(lengths))]
                ctx.trajectory_end_idx = [t - 1 for t in ctx.trajectory_end_idx]
                self._reset_transitions()
                break
