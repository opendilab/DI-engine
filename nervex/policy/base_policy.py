from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch
from nervex.worker import TransitionBuffer


class Policy(ABC):
    learn_function = namedtuple('learn_function', ['data_preprocess', 'forward'])
    collect_function = namedtuple(
        'collect_function', [
            'data_preprocess', 'forward', 'data_postprocess', 'process_transition', 'get_trajectory',
            'callback_episode_done'
        ]
    )
    eval_function = namedtuple(
        'collect_function', ['data_preprocess', 'forward', 'data_postprocess', 'callback_episode_done']
    )
    control_function = namedtuple('control_function', ['get_setting_learn', 'get_setting_collect', 'get_setting_eval'])

    def __init__(
            self, cfg: dict, model: Optional[torch.nn.Module] = None, enable_field: Optional[List[str]] = None
    ) -> None:
        if model is None:
            # create model from cfg
            model = self._create_model_from_cfg(cfg)
        self._cfg = cfg
        self._use_cuda = cfg.use_cuda
        if self._use_cuda:
            model.cuda()
        self._model = model
        self._enable_field = enable_field
        self._total_field = set(['learn', 'collect', 'eval', 'control'])
        if self._enable_field is None:
            self._init_learn()
            self._init_collect()
            self._init_eval()
            self._init_control()
        else:
            assert set(self._enable_field).issubset(self._total_field), self._enable_field
            for field in self._enable_field:
                getattr(self, '_init_' + field)()

    @abstractmethod
    def _create_model_from_cfg(self, cfg: dict) -> torch.nn.Module:
        raise NotImplementedError

    @abstractmethod
    def _init_learn(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _init_collect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _init_eval(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _init_control(self) -> None:
        raise NotImplementedError

    @property
    def learn(self) -> 'Policy.learn_function':  # noqa
        return Policy.learn_function(self._data_preprocess_learn, self._forward_learn)

    @property
    def collect(self) -> 'Policy.collect_function':  # noqa
        return Policy.collect_function(
            self._data_preprocess_collect, self._forward_collect, self._data_postprocess_collect,
            self._process_transition, self._get_trajectory, self._callback_episode_done_collect
        )

    @property
    def eval(self) -> 'Policy.eval_function':  # noqa
        return Policy.eval_function(
            self._data_preprocess_collect, self._forward_eval, self._data_postprocess_collect,
            self._callback_episode_done_collect
        )

    @property
    def control(self) -> 'Policy.control_function':  # noqa
        return Policy.control_function(self._get_setting_learn, self._get_setting_collect, self._get_setting_eval)

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    # *************************************** learn function ************************************
    @abstractmethod
    def _data_preprocess_learn(self, data: List[Any]) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        raise NotImplementedError

    # *************************************** collect function ************************************

    @abstractmethod
    def _data_preprocess_collect(self, data: Dict[int, Any]) -> Tuple[List[int], dict]:
        raise NotImplementedError

    @abstractmethod
    def _forward_collect(self, data: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _data_postprocess_collect(self, data_id: List[int], data: dict) -> Dict[int, dict]:
        raise NotImplementedError

    @abstractmethod
    def _process_transition(self, obs: Any, agent_output: dict, timestep: namedtuple) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _get_trajectory(self, transitions: TransitionBuffer, data_id: int, done: bool) -> Union[None, List[Any]]:
        raise NotImplementedError

    @abstractmethod
    def _callback_episode_done_collect(self, data_id: int) -> None:
        raise NotImplementedError

    # *************************************** eval function ************************************

    @abstractmethod
    def _forward_eval(self, data: dict) -> Dict[str, Any]:
        raise NotImplementedError

    # *************************************** control function ************************************
    @abstractmethod
    def _get_setting_learn(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _get_setting_collect(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _get_setting_eval(self) -> dict:
        raise NotImplementedError
