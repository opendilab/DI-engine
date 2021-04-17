from abc import ABC, abstractmethod
from collections import namedtuple, deque
from typing import Optional, List, Dict, Any, Tuple, Union

import torch
from easydict import EasyDict

from nervex.model import create_model
from nervex.utils import import_module, allreduce, broadcast, get_rank, POLICY_REGISTRY


class Policy(ABC):
    learn_function = namedtuple(
        'learn_function', [
            'data_preprocess',
            'forward',
            'reset',
            'info',
            'monitor_vars',
            'get_attribute',
            'set_attribute',
            'state_dict',
            'load_state_dict',
        ]
    )
    collect_function = namedtuple(
        'collect_function', [
            'data_preprocess',
            'forward',
            'data_postprocess',
            'process_transition',
            'get_train_sample',
            'reset',
            'get_attribute',
            'set_attribute',
            'state_dict',
            'load_state_dict',
        ]
    )
    eval_function = namedtuple(
        'eval_function', [
            'data_preprocess',
            'forward',
            'data_postprocess',
            'reset',
            'get_attribute',
            'set_attribute',
            'state_dict',
            'load_state_dict',
        ]
    )
    total_field = set(['learn', 'collect', 'eval'])

    def __init__(
            self,
            cfg: dict,
            model: Optional[Union[type, torch.nn.Module]] = None,
            enable_field: Optional[List[str]] = None
    ) -> None:
        self._cfg = cfg
        model = self._create_model(cfg, model)
        self._use_cuda = cfg.use_cuda and torch.cuda.is_available()
        self._use_distributed = cfg.get('use_distributed', False)
        self._rank = get_rank() if self._use_distributed else 0
        if self._use_distributed:
            self._init_multi_gpu_setting(model)
        self._device = 'cuda:{}'.format(self._rank % torch.cuda.device_count()) if self._use_cuda else 'cpu'
        if self._use_cuda:
            torch.cuda.set_device(self._rank)
            model.cuda()
        self._model = model

        if enable_field is None:
            self._enable_field = self.total_field
        else:
            self._enable_field = enable_field
        assert set(self._enable_field).issubset(self.total_field), self._enable_field
        for field in self._enable_field:
            getattr(self, '_init_' + field)()

    def _init_multi_gpu_setting(self, model: torch.nn.Module) -> None:
        for name, param in model.state_dict().items():
            assert isinstance(param.data, torch.Tensor), type(param.data)
            broadcast(param.data, 0)
        for name, param in model.named_parameters():
            setattr(param, 'grad', torch.zeros_like(param))

    def _create_model(self, cfg: dict, model: Optional[torch.nn.Module] = None) -> torch.nn.Module:
        model_cfg = cfg.model
        if model is None:
            if 'model_type' not in model_cfg:
                model_type, import_names = self.default_model()
                model_cfg.model_type = model_type
                model_cfg.import_names = import_names
            return create_model(model_cfg)
        else:
            if isinstance(model, torch.nn.Module):
                return model
            else:
                raise RuntimeError("invalid model: {}".format(type(model)))

    @abstractmethod
    def _init_learn(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _init_collect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _init_eval(self) -> None:
        raise NotImplementedError

    @property
    def learn_mode(self) -> 'Policy.learn_function':  # noqa
        return Policy.learn_function(
            self._data_preprocess_learn,
            self._forward_learn,
            self._reset_learn,
            self.__repr__,
            self._monitor_vars_learn,
            self._get_attribute,
            self._set_attribute,
            self._state_dict_learn,
            self._load_state_dict_learn,
        )

    @property
    def collect_mode(self) -> 'Policy.collect_function':  # noqa
        return Policy.collect_function(
            self._data_preprocess_collect,
            self._forward_collect,
            self._data_postprocess_collect,
            self._process_transition,
            self._get_train_sample,
            self._reset_collect,
            self._get_attribute,
            self._set_attribute,
            self._state_dict_collect,
            self._load_state_dict_collect,
        )

    @property
    def eval_mode(self) -> 'Policy.eval_function':  # noqa
        return Policy.eval_function(
            self._data_preprocess_collect,
            self._forward_eval,
            self._data_postprocess_collect,
            self._reset_eval,
            self._get_attribute,
            self._set_attribute,
            self._state_dict_eval,
            self._load_state_dict_eval,
        )

    def _set_attribute(self, name: str, value: Any) -> None:
        setattr(self, '_' + name, value)

    def _get_attribute(self, name: str) -> Any:
        attributes = ['batch_size', 'use_cuda', 'device']
        assert name in attributes, 'attr<{}> not in {}'.format(name, attributes)
        if hasattr(self, '_get_' + name):
            return getattr(self, '_get_' + name)()
        elif hasattr(self, '_' + name):
            return getattr(self, '_' + name)
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        return "nerveX DRL Policy\n{}".format(repr(self._model))

    def sync_gradients(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                allreduce(param.grad.data)

    # don't need to implement default_model method by default
    def default_model(self) -> Tuple[str, List[str]]:
        raise NotImplementedError

    # *************************************** learn function ************************************
    @abstractmethod
    def _data_preprocess_learn(self, data: List[Any]) -> Tuple[dict, dict]:
        raise NotImplementedError

    @abstractmethod
    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
        raise NotImplementedError

    def _monitor_vars_learn(self) -> List[str]:
        return ['cur_lr', 'total_loss']

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {'model': self._model.state_dict()}

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._model.load_state_dict(state_dict['model'], strict=True)

    def _get_batch_size(self) -> Union[int, Dict[str, int]]:
        return self._cfg.learn.batch_size

    # *************************************** collect function ************************************

    @abstractmethod
    def _data_preprocess_collect(self, data: Dict[int, Any]) -> Tuple[List[int], dict]:
        raise NotImplementedError

    @abstractmethod
    def _forward_collect(self, data_id: List[int], data: dict, **kwargs) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _data_postprocess_collect(self, data_id: List[int], data: dict) -> Dict[int, dict]:
        raise NotImplementedError

    @abstractmethod
    def _process_transition(self, obs: Any, armor_output: dict, timestep: namedtuple) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _get_train_sample(self, data: deque) -> Union[None, List[Any]]:
        raise NotImplementedError

    @abstractmethod
    def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
        raise NotImplementedError

    def _state_dict_collect(self) -> Dict[str, Any]:
        return {'model': self._model.state_dict()}

    def _load_state_dict_collect(self, state_dict: Dict[str, Any]) -> None:
        self._model.load_state_dict(state_dict['model'], strict=True)

    # *************************************** eval function ************************************

    @abstractmethod
    def _forward_eval(self, data_id: List[int], data: dict) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        raise NotImplementedError

    def _state_dict_eval(self) -> Dict[str, Any]:
        return {'model': self._model.state_dict()}

    def _load_state_dict_eval(self, state_dict: Dict[str, Any]) -> None:
        self._model.load_state_dict(state_dict['model'], strict=True)


class CommandModePolicy(Policy):
    command_function = namedtuple('command_function', ['get_setting_learn', 'get_setting_collect', 'get_setting_eval'])
    total_field = set(['learn', 'collect', 'eval', 'command'])

    @property
    def command_mode(self) -> 'Policy.command_function':  # noqa
        return CommandModePolicy.command_function(
            self._get_setting_learn, self._get_setting_collect, self._get_setting_eval
        )

    @abstractmethod
    def _init_command(self) -> None:
        raise NotImplementedError

    # *************************************** command function ************************************
    @abstractmethod
    def _get_setting_learn(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _get_setting_collect(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _get_setting_eval(self) -> dict:
        raise NotImplementedError


def create_policy(cfg: dict, **kwargs) -> Policy:
    cfg = EasyDict(cfg)
    import_module(cfg.get('import_names', []))
    return POLICY_REGISTRY.build(cfg.policy_type, cfg=cfg, **kwargs)
