from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Optional, List, Dict, Any, Tuple, Union

import torch
import copy
from easydict import EasyDict

from ding.model import create_model
from ding.utils import import_module, allreduce, broadcast, get_rank, allreduce_async, synchronize, POLICY_REGISTRY


class Policy(ABC):

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    learn_function = namedtuple(
        'learn_function', [
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
            'forward',
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
            'forward',
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
        self._on_policy = self._cfg.on_policy
        if enable_field is None:
            self._enable_field = self.total_field
        else:
            self._enable_field = enable_field
        assert set(self._enable_field).issubset(self.total_field), self._enable_field

        if len(set(self._enable_field).intersection(set(['learn', 'collect', 'eval']))) > 0:
            model = self._create_model(cfg, model)
            self._cuda = cfg.cuda and torch.cuda.is_available()
            # now only support multi-gpu for only enable learn mode
            if len(set(self._enable_field).intersection(set(['learn']))) > 0:
                self._rank = get_rank() if self._cfg.learn.multi_gpu else 0
                if self._cuda:
                    torch.cuda.set_device(self._rank % torch.cuda.device_count())
                    model.cuda()
                if self._cfg.learn.multi_gpu:
                    bp_update_sync = self._cfg.learn.get('bp_update_sync', True)
                    self._bp_update_sync = bp_update_sync
                    self._init_multi_gpu_setting(model, bp_update_sync)
            else:
                self._rank = 0
                if self._cuda:
                    torch.cuda.set_device(self._rank % torch.cuda.device_count())
                    model.cuda()
            self._model = model
            self._device = 'cuda:{}'.format(self._rank % torch.cuda.device_count()) if self._cuda else 'cpu'
        else:
            self._cuda = False
            self._rank = 0
            self._device = 'cpu'

        for field in self._enable_field:
            getattr(self, '_init_' + field)()

    def _init_multi_gpu_setting(self, model: torch.nn.Module, bp_update_sync: bool) -> None:
        for name, param in model.state_dict().items():
            assert isinstance(param.data, torch.Tensor), type(param.data)
            broadcast(param.data, 0)
        for name, param in model.named_parameters():
            setattr(param, 'grad', torch.zeros_like(param))
        if not bp_update_sync:

            def make_hook(name, p):

                def hook(*ignore):
                    allreduce_async(name, p.grad.data)

                return hook

            for i, (name, p) in enumerate(model.named_parameters()):
                if p.requires_grad:
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(make_hook(name, p))

    def _create_model(self, cfg: dict, model: Optional[torch.nn.Module] = None) -> torch.nn.Module:
        if model is None:
            model_cfg = cfg.model
            if 'type' not in model_cfg:
                m_type, import_names = self.default_model()
                model_cfg.type = m_type
                model_cfg.import_names = import_names
            return create_model(model_cfg)
        else:
            if isinstance(model, torch.nn.Module):
                return model
            else:
                raise RuntimeError("invalid model: {}".format(type(model)))

    @property
    def cfg(self) -> EasyDict:
        return self._cfg

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
            self._forward_collect,
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
            self._forward_eval,
            self._reset_eval,
            self._get_attribute,
            self._set_attribute,
            self._state_dict_eval,
            self._load_state_dict_eval,
        )

    def _set_attribute(self, name: str, value: Any) -> None:
        setattr(self, '_' + name, value)

    def _get_attribute(self, name: str) -> Any:
        if hasattr(self, '_get_' + name):
            return getattr(self, '_get_' + name)()
        elif hasattr(self, '_' + name):
            return getattr(self, '_' + name)
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        return "DI-engine DRL Policy\n{}".format(repr(self._model))

    def sync_gradients(self, model: torch.nn.Module) -> None:
        if self._bp_update_sync:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    allreduce(param.grad.data)
        else:
            synchronize()

    # don't need to implement default_model method by force
    def default_model(self) -> Tuple[str, List[str]]:
        raise NotImplementedError

    # *************************************** learn function ************************************

    @abstractmethod
    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        raise NotImplementedError

    # don't need to implement _reset_learn method by force
    def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
        pass

    def _monitor_vars_learn(self) -> List[str]:
        return ['cur_lr', 'total_loss']

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {'model': self._learn_model.state_dict()}

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'], strict=True)

    def _get_batch_size(self) -> Union[int, Dict[str, int]]:
        return self._cfg.learn.batch_size

    # *************************************** collect function ************************************

    @abstractmethod
    def _forward_collect(self, data: dict, **kwargs) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        raise NotImplementedError

    # don't need to implement _reset_collect method by force
    def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
        pass

    def _state_dict_collect(self) -> Dict[str, Any]:
        return {'model': self._collect_model.state_dict()}

    def _load_state_dict_collect(self, state_dict: Dict[str, Any]) -> None:
        self._collect_model.load_state_dict(state_dict['model'], strict=True)

    # *************************************** eval function ************************************

    @abstractmethod
    def _forward_eval(self, data: dict) -> Dict[str, Any]:
        raise NotImplementedError

    # don't need to implement _reset_eval method by force
    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        pass

    def _state_dict_eval(self) -> Dict[str, Any]:
        return {'model': self._eval_model.state_dict()}

    def _load_state_dict_eval(self, state_dict: Dict[str, Any]) -> None:
        self._eval_model.load_state_dict(state_dict['model'], strict=True)


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
    def _get_setting_learn(self, command_info: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _get_setting_collect(self, command_info: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _get_setting_eval(self, command_info: dict) -> dict:
        raise NotImplementedError


def create_policy(cfg: dict, **kwargs) -> Policy:
    cfg = EasyDict(cfg)
    import_module(cfg.get('import_names', []))
    return POLICY_REGISTRY.build(cfg.type, cfg=cfg, **kwargs)


def get_policy_cls(cfg: EasyDict) -> type:
    import_module(cfg.get('import_names', []))
    return POLICY_REGISTRY.get(cfg.type)
