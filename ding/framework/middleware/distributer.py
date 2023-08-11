import numpy as np
from time import sleep, time
from dataclasses import fields
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Union
from ditk import logging
from ding.framework import task
from ding.data import StorageLoader, Storage, ModelLoader
if TYPE_CHECKING:
    from ding.framework.context import Context
    from torch.nn import Module


class ContextExchanger:

    def __init__(self, skip_n_iter: int = 1, storage_loader: Optional[StorageLoader] = None) -> None:
        """
        Overview:
            Exchange context between processes,
            support properties: trajectories, episodes, env_step, env_episode, train_iter
        Arguments:
            - skip_n_iter (:obj:`int`): For collectors, it may be necessary to skip waiting \
                for the first n iterations to collect data for the learner to learn. This parameter \
                will not work on learner.
            - storage_loader (:obj:`Optional[StorageLoader]`): Turn data into storage class to reduce \
                the network overhead.
        """
        if not task.router.is_active:
            raise RuntimeError("ContextHandler should be used in parallel mode!")
        self._state = {}
        self._local_state = {}  # just save local state, not send to remote node
        if task.has_role(task.role.COLLECTOR):
            self._local_state['env_step'] = 0
            self._local_state['env_episode'] = 0
        self._event_name = "context_exchanger_{role}"
        self._skip_n_iter = skip_n_iter
        self._storage_loader = storage_loader
        for role in task.role:  # Only subscribe to other roles
            if not task.has_role(role):
                task.on(self._event_name.format(role=role), self.put)
        if storage_loader:
            task.once("finish", lambda _: storage_loader.shutdown())

    def __new__(cls, *args, **kwargs):
        if not task.router.is_active:
            return task.void()

        if len(task.roles) == 0:
            logging.warning("The task does not have any roles defined, the ContextExchanger will not work.")
            return task.void()

        if len(task.roles) > 1:
            logging.warning(
                "Use multiple roles in one exchanger may lead to unexpected result, please check your code."
            )

        return super(ContextExchanger, cls).__new__(cls)

    def __call__(self, ctx: "Context"):
        self.merge(ctx)
        yield
        payload = self.fetch(ctx)
        if payload:
            if self._storage_loader and task.has_role(task.role.COLLECTOR):
                payload = self._storage_loader.save(payload)
            for role in task.roles:
                task.emit(self._event_name.format(role=role), payload, only_remote=True)

    def __del__(self):
        if self._storage_loader:
            self._storage_loader.shutdown()

    def put(self, payload: Union[Dict, Storage]):
        """
        Overview:
            Get attributes from ctx on the callback of event.
            Each attribute should have a standalone put handler, which named `_put_{key}`
        """

        def callback(payload: Dict):
            for key, item in payload.items():
                fn_name = "_put_{}".format(key)
                if hasattr(self, fn_name):
                    getattr(self, fn_name)(item)
                else:
                    logging.warning("Receive unexpected key ({}) in context exchanger".format(key))

        if isinstance(payload, Storage):
            assert self._storage_loader is not None, "Storage loader is not defined when data is a storage object."
            self._storage_loader.load(payload, callback)
        else:
            callback(payload)

    def fetch(self, ctx: "Context") -> Dict[str, Any]:
        """
        Overview:
            Fetch attributes from ctx before emit them to the event bus.
            Each attribute should have a standalone fetch handler, which named `_fetch_{key}`
        """
        payload = {}
        for field in fields(ctx):
            key, item = field.name, getattr(ctx, field.name)
            fn_name = "_fetch_{}".format(key)
            if hasattr(self, fn_name):
                value = getattr(self, fn_name)(item)
                if value is not None:
                    payload[key] = value
        return payload

    def merge(self, ctx: "Context"):
        if task.has_role(task.role.LEARNER):
            # Learner should always wait for trajs.
            # TODO: Automaticlly wait based on properties, not roles.
            while len(self._state) == 0:
                sleep(0.01)
        elif ctx.total_step >= self._skip_n_iter:
            start = time()
            while len(self._state) == 0:
                if time() - start > 60:
                    logging.warning("Timeout when waiting for new context! Node id: {}".format(task.router.node_id))
                    break
                sleep(0.01)

        for k, v in self._state.items():
            if not task.has_role(task.role.COLLECTOR) and k.startswith('increment_'):
                pure_k = k.split('increment_')[-1]
                setattr(ctx, pure_k, getattr(ctx, pure_k) + v)
            else:
                setattr(ctx, k, v)
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

    def _put_env_step(self, increment_env_step: int):
        if not task.has_role(task.role.COLLECTOR):
            if 'increment_env_step' not in self._state:
                self._state['increment_env_step'] = 0
            self._state["increment_env_step"] += increment_env_step

    def _fetch_env_step(self, env_step: int):
        if task.has_role(task.role.COLLECTOR):
            increment_env_step = env_step - self._local_state['env_step']
            self._local_state['env_step'] = env_step
            return increment_env_step

    def _put_env_episode(self, increment_env_episode: int):
        if not task.has_role(task.role.COLLECTOR):
            if 'increment_env_episode' not in self._state:
                self._state['increment_env_episode'] = 0
            self._state["increment_env_episode"] += increment_env_episode

    def _fetch_env_episode(self, env_episode: int):
        if task.has_role(task.role.COLLECTOR):
            increment_env_episode = env_episode - self._local_state['env_episode']
            self._local_state['env_episode'] = env_episode
            return increment_env_episode

    def _put_train_iter(self, train_iter: int):
        if not task.has_role(task.role.LEARNER):
            self._state["train_iter"] = train_iter

    def _fetch_train_iter(self, train_iter: int):
        if task.has_role(task.role.LEARNER):
            return train_iter


class ModelExchanger:

    def __init__(self, model: "Module", model_loader: Optional[ModelLoader] = None) -> None:
        """
        Overview:
            Exchange model between processes, only the learner will send the model,
            otherwise the model will only be received.
            If you are using a shared model on a single host, there is no need to use this middleware.
        Arguments:
            - model (:obj:`torch.nn.Module`): Pytorch module.
            - model_loader (:obj:`ModelLoader`): Encode model in subprocess.
        """
        self._model = model
        self._model_loader = model_loader
        self._event_name = "model_exchanger"
        self._state_dict_cache: Optional[Union[object, Storage]] = None
        self._is_learner = task.has_role(task.role.LEARNER)
        if not self._is_learner:
            task.on(self._event_name, self._cache_state_dict)
        if model_loader:
            task.once("finish", lambda _: model_loader.shutdown())

    def _cache_state_dict(self, state_dict: Union[object, Storage]):
        self._state_dict_cache = state_dict

    def __new__(cls, *args, **kwargs):
        if not task.router.is_active:
            return task.void()

        if len(task.roles) == 0:
            logging.warning("The task does not have any roles defined, the ModelExchanger will not work.")
            return task.void()

        if len(task.roles) > 1:
            logging.warning(
                "Use multiple roles in one exchanger may lead to unexpected result, please check your code."
            )

        return super(ModelExchanger, cls).__new__(cls)

    def __call__(self, ctx: "Context") -> Any:
        if self._model_loader:
            self._model_loader.start()

        if not self._is_learner:
            if ctx.total_step != 0:  # Skip first iteration
                self._update_model()
        else:
            yield
            self._send_model()

    def _update_model(self):
        start = time()
        while True:
            if task.finish:
                return
            if time() - start > 60:
                logging.warning("Timeout when waiting for new model! Node id: {}".format(task.router.node_id))
                break
            if self._state_dict_cache is None:
                sleep(0.01)
            else:
                if isinstance(self._state_dict_cache, Storage) and self._model_loader is not None:
                    try:
                        self._model.load_state_dict(self._model_loader.load(self._state_dict_cache))
                        self._state_dict_cache = None
                        break
                    except FileNotFoundError as e:
                        logging.warning(
                            "Model file has been deleted on node {}, maybe you can increase the ttl.".format(
                                task.router.node_id
                            )
                        )
                        self._state_dict_cache = None
                        continue
                else:
                    self._model.load_state_dict(self._state_dict_cache)
                    self._state_dict_cache = None
                    break

    def _send_model(self):
        if self._model_loader:
            self._model_loader.save(self._send_callback)
        else:
            task.emit(self._event_name, self._model.state_dict(), only_remote=True)

    def _send_callback(self, storage: Storage):
        if task.running:
            task.emit(self._event_name, storage, only_remote=True)

    def __del__(self):
        if self._model_loader:
            self._model_loader.shutdown()


class PeriodicalModelExchanger:

    def __init__(
            self,
            model: "Module",
            mode: str,
            period: int = 1,
            delay_toleration: float = np.inf,
            stale_toleration: int = 1,
            event_name: str = "model_exchanger",
            model_loader: Optional[ModelLoader] = None
    ) -> None:
        """
        Overview:
            Exchange model between processes, set the mode to "send" or "receive" to specify the role of the process.
            If you are using a shared model on a single host, there is no need to use this middleware.
        Arguments:
            - model (:obj:`torch.nn.Module`): Pytorch module.
            - mode (:obj:`str`): "send" or "receive".
            - period (:obj:`int`): The period of model exchange.
            - delay_toleration (:obj:`float`): The permitted time interval for receiving model after being sent.
            - stale_toleration (:obj:`int`): The permitted number of iterations for receiving model after being sent.
            - event_name (:obj:`str`): The event name for model exchange.
            - model_loader (:obj:`ModelLoader`): ModelLoader for this PeriodicalModelExchanger to use.
        """
        self._model = model
        self._model_loader = model_loader
        self._event_name = event_name
        self._period = period
        self._mode = mode
        if self._mode == "receive":
            self._id_counter = -1
            self._model_id = -1
        else:
            self._id_counter = 0
        self._stale_toleration = stale_toleration
        self._model_stale = stale_toleration
        self._delay_toleration = delay_toleration
        self._state_dict_cache: Optional[Union[object, Storage]] = None

        if self._mode == "receive":
            task.on(self._event_name, self._cache_state_dict)
        if model_loader:
            task.once("finish", lambda _: model_loader.shutdown())

    def _cache_state_dict(self, msg: Dict[str, Any]):
        if msg['id'] % self._period == 0:
            self._state_dict_cache = msg['model']
            self._id_counter = msg['id']
            self._time = msg['time']

    def __new__(cls, *args, **kwargs):
        return super(PeriodicalModelExchanger, cls).__new__(cls)

    def __call__(self, ctx: "Context") -> Any:
        if self._model_loader:
            self._model_loader.start()

        if self._mode == "receive":
            if ctx.total_step != 0:  # Skip first iteration
                self._update_model()
        elif self._mode == "send":
            yield
            if self._id_counter % self._period == 0:
                self._send_model(id=self._id_counter)
            self._id_counter += 1
        else:
            raise NotImplementedError

    def _update_model(self):
        start = time()
        while True:
            if task.finish:
                return
            if time() - start > 60:
                logging.warning("Timeout when waiting for new model! Node id: {}".format(task.router.node_id))
                self._model_stale += 1
                break
            if self._state_dict_cache is None:
                if self._model_stale < self._stale_toleration and time() - self._time < self._delay_toleration:
                    self._model_stale += 1
                    break
                else:
                    sleep(0.01)
            else:
                if self._id_counter > self._model_id and time() - self._time < self._delay_toleration:
                    if isinstance(self._state_dict_cache, Storage) and self._model_loader is not None:
                        try:
                            self._model.load_state_dict(self._model_loader.load(self._state_dict_cache))
                            self._state_dict_cache = None
                            self._model_id = self._id_counter
                            self._model_stale = 1
                            break
                        except FileNotFoundError as e:
                            logging.warning(
                                "Model file has been deleted on node {}, maybe you can increase the ttl.".format(
                                    task.router.node_id
                                )
                            )
                            self._state_dict_cache = None
                            continue
                    else:
                        self._model.load_state_dict(self._state_dict_cache)
                        self._state_dict_cache = None
                        self._model_id = self._id_counter
                        self._model_stale = 1
                        break
                else:
                    self._model_stale += 1

    def _send_model(self, id: int):
        if self._model_loader:
            self._model_loader.save(self._send_callback)
        else:
            task.emit(self._event_name, {'id': id, 'model': self._model.state_dict(), 'time': time()}, only_remote=True)

    def _send_callback(self, storage: Storage):
        if task.running:
            task.emit(self._event_name, storage, only_remote=True)

    def __del__(self):
        if self._model_loader:
            self._model_loader.shutdown()
