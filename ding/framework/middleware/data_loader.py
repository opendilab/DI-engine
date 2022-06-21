from typing import TYPE_CHECKING, Any, List
from ding.framework import task
from ditk import logging
from ding.data import Storage, FileStorage

if TYPE_CHECKING:
    from ding.framework import OnlineRLContext


class DataLoader:

    def __new__(cls, *args, **kwargs):
        if not task.router.is_active:
            logging.warning("DataLoader can only be used in parallel mode, please use ditask to start the task.")
            return task.void()
        if not (task.has_role(task.role.COLLECTOR) or task.has_role(task.role.LEARNER)):
            logging.warning("DataLoader can only be used in collector or learner.")
            return task.void()
        return super(DataLoader, cls).__new__(cls)

    def __init__(self) -> None:
        pass

    def __call__(self, ctx: "OnlineRLContext") -> Any:
        if task.has_role(task.role.LEARNER):
            if ctx.trajectories is not None:
                ctx.trajectories = self._load_data(ctx.trajectories)
            if ctx.episodes is not None:
                ctx.episodes = self._load_data(ctx.trajectories)
        yield
        if task.has_role(task.role.COLLECTOR):
            if ctx.trajectories is not None:
                ctx.trajectories = self._encode_data(ctx.trajectories)
            if ctx.episodes is not None:
                ctx.episodes = self._encode_data(ctx.trajectories)

    def _load_data(self, traj: List[Any]) -> List[Any]:
        # print("LOAD TRAJ", traj)
        return

    def _encode_data(self, traj: List[Any]) -> List[Any]:
        # print("ENCODE TRAJ", traj)
        return
