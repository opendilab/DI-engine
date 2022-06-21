import sys
from threading import Thread
import pytest
import torch
import gym
from ding.framework import Parallel, task, OnlineRLContext
from ding.framework.middleware import DataLoader
from ding.envs.env_manager import EnvSupervisor
from ding.envs import DingEnvWrapper
from ding.framework.supervisor import ChildType


def data_loader_main():

    def step_collector(ctx: OnlineRLContext):
        # Mock step collector, set obs bigger than 10MB (1024, 1024, 10)
        ctx.trajectories = [{"s": "abc", "obs": torch.rand(1024, 1024)} for _ in range(20)]
        print("SIZE OF", sys.getsizeof(ctx.trajectories))

    with task.start(ctx=OnlineRLContext()):
        if task.router.node_id == 0:
            task.add_role(task.role.LEARNER)
        else:
            task.add_role(task.role.COLLECTOR)

        task.use(DataLoader())

        if task.has_role(task.role.COLLECTOR):
            task.use(step_collector)

        task.run(1)


@pytest.mark.unittest
def test_data_loader():
    Parallel.runner(n_parallel_workers=2)(data_loader_main)
