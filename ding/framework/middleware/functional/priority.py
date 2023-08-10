from typing import TYPE_CHECKING, Callable
from ding.framework import task
if TYPE_CHECKING:
    from ding.framework import OnlineRLContext


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
