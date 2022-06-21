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
    with task.start():
        if task.router.node_id == 0:
            task.add_role(task.role.LEARNER)
        else:

            def step_collector(ctx: OnlineRLContext):
                # Mock step collector, set obs bigger than 10MB (1024, 1024, 10)
                ctx.trajectories = [{"s": "abc", "obs": torch.rand(1024, 1024)} for _ in range(20)]

        task.use(DataLoader())


@pytest.mark.unittest
def test_data_loader():
    Parallel.runner(n_parallel_workers=2)(data_loader_main)
