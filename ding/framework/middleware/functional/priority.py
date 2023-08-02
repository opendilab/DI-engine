from typing import Callable
import torch
from ding.framework import task
from ding.framework import OnlineRLContext


def priority_calculator(func_for_priority_calculation: Callable) -> Callable:
    """
    Overview:
        The middleware that calculates the priority of the collected data.
    Arguments:
        - func_for_priority_calculation (:obj:`Callable`): The function that calculates \
            the priority of the collected data.
    """

    if task.router.is_active and not task.has_role(task.role.COLLECTOR):
        return task.void()

    def _priority_calculator(ctx: "OnlineRLContext") -> None:

        priority = func_for_priority_calculation(ctx.trajectories)
        for i in range(len(priority)):
            ctx.trajectories[i]['priority'] = torch.tensor(priority[i], dtype=torch.float32).unsqueeze(-1)

    return _priority_calculator
