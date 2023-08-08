from typing import TYPE_CHECKING, Optional, Callable, Dict, List, Union
from ditk import logging
from easydict import EasyDict
from matplotlib import pyplot as plt
from matplotlib import animation
import os
import numpy as np
import torch
import wandb
import pickle
import treetensor.numpy as tnp
from ding.policy import Policy
from ding.data import Buffer
from ding.rl_utils import gae, gae_data
from ding.framework import task
from ding.utils.data import ttorch_collate
from ding.torch_utils import to_device


def priority_calculator(priority_calculation_fn: Callable) -> Callable:
    """
    Overview:
        The middleware that calculates the priority of the collected data.
    Arguments:
        - priority_calculation_fn (:obj:`Callable`): The function that calculates the priority of the collected data.
    """

    if task.router.is_active and not task.has_role(task.role.COLLECTOR):
        return task.void()

    def _priority_calculator(ctx: "OnlineRLContext") -> None:

        priority = priority_calculation_fn(ctx.trajectories)
        for i in range(len(priority)):
            ctx.trajectories[i]['priority'] = priority[i]

    return _priority_calculator
