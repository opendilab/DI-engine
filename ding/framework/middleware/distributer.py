from time import sleep
from typing import TYPE_CHECKING, List, Dict, Any
from ditk import logging
from ding.framework import task
if TYPE_CHECKING:
    from ding.framework.context import Context


class ContextExchanger:
    """
    Overview:
        Exchange context between processes,
        support properties: trajectories, episodes, env_step, env_episode, train_iter
    """

    def __init__(self, skip_n_iter: int = 1) -> None:
        if not task.router.is_active:
            raise RuntimeError("ContextHandler should be used in parallel mode!")
        self._state = {}
        self._event_name = "context_exchanger"
        self._skip_n_iter = skip_n_iter
        task.on(self._event_name, self.put)

    def __new__(cls, *args, **kwargs):
        if not task.router.is_active:
            return task.void()

        if len(task.roles) == 0:
            logging.warning("The task does not have any roles defined, the ContextExchanger will not work.")
            return task.void()

        return super(ContextExchanger, cls).__new__(cls)

    def __call__(self, ctx: "Context"):
        self.merge(ctx)
        yield
        payload = self.fetch(ctx)
        if payload:
            task.emit(self._event_name, payload, only_remote=True)

    def put(self, ctx_payload: Dict):
        """
        Overview:
            Get attributes from ctx on the callback of event.
            Each attribute should have a standalone put handler, which named `_put_{key}`
        """
        for key, item in ctx_payload.items():
            fn_name = "_put_{}".format(key)
            if hasattr(self, fn_name):
                getattr(self, fn_name)(item)
            else:
                logging.warning("Receive unexpected key ({}) in context exchanger".format(key))

    def fetch(self, ctx: "Context") -> Dict[str, Any]:
        """
        Overview:
            Fetch attributes from ctx before emit them to the event bus.
            Each attribute should have a standalone fetch handler, which named `_fetch_{key}`
        """
        payload = {}
        for key, item in ctx.items():
            fn_name = "_fetch_{}".format(key)
            if hasattr(self, fn_name):
                value = getattr(self, fn_name)(item)
                if value is not None:
                    payload[key] = value
        return payload

    def merge(self, ctx: "Context"):
        if ctx.total_step >= self._skip_n_iter or task.has_role(task.role.LEARNER):
            while len(self._state) == 0:
                sleep(0.01)
        for k, v in self._state.items():
            ctx[k] = v
        self._state = {}

    # Handle each attibute of context
    def _put_trajectories(self, traj: List[Any]):
        if not task.has_role(task.role.LEARNER):
            return
        if "trajectories" not in self._state:
            self._state["trajectories"] = []
        self._state["trajectories"].extend(traj)

    def _fetch_trajectories(self, traj: List[Any]):
        if task.has_role(task.role.COLLECTOR):
            return traj

    def _put_episodes(self, episodes: List[Any]):
        if not task.has_role(task.role.LEARNER):
            return
        if "episodes" not in self._state:
            self._state["episodes"] = []
        self._state["episodes"].extend(episodes)

    def _fetch_episodes(self, episodes: List[Any]):
        if task.has_role(task.role.COLLECTOR):
            return episodes

    def _put_trajectory_end_idx(self, trajectory_end_idx: List[str]):
        if not task.has_role(task.role.LEARNER):
            return
        if "trajectory_end_idx" not in self._state:
            self._state["trajectory_end_idx"] = []
        self._state["trajectory_end_idx"].extend(trajectory_end_idx)

    def _fetch_trajectory_end_idx(self, trajectory_end_idx: List[str]):
        if task.has_role(task.role.COLLECTOR):
            return trajectory_end_idx

    def _put_env_step(self, env_step: int):
        if not task.has_role(task.role.COLLECTOR):
            self._state["env_step"] = env_step

    def _fetch_env_step(self, env_step: int):
        if task.has_role(task.role.COLLECTOR):
            return env_step

    def _put_env_episode(self, env_episode: int):
        if not task.has_role(task.role.COLLECTOR):
            self._state["env_episode"] = env_episode

    def _fetch_env_episode(self, env_episode: int):
        if task.has_role(task.role.COLLECTOR):
            return env_episode

    def _put_train_iter(self, train_iter: int):
        if not task.has_role(task.role.LEARNER):
            self._state["train_iter"] = train_iter

    def _fetch_train_iter(self, train_iter: int):
        if task.has_role(task.role.LEARNER):
            return train_iter
