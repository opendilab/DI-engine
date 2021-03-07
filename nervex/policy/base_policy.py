from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple, Union
from collections import namedtuple, deque
from easydict import EasyDict
import torch

from nervex.model import create_model
from nervex.utils import import_module, allreduce, broadcast, get_rank


class Policy(ABC):
    learn_function = namedtuple(
        'learn_function', [
            'data_preprocess', 'forward', 'reset', 'info', 'state_dict_handle', 'set_setting', 'monitor_vars',
            'get_batch_size'
        ]
    )
    collect_function = namedtuple(
        'collect_function', [
            'data_preprocess', 'forward', 'data_postprocess', 'process_transition', 'get_train_sample', 'reset',
            'set_setting', 'state_dict_handle'
        ]
    )
    eval_function = namedtuple(
        'eval_function',
        ['data_preprocess', 'forward', 'data_postprocess', 'reset', 'set_setting', 'state_dict_handle']
    )
    command_function = namedtuple('command_function', ['get_setting_learn', 'get_setting_collect', 'get_setting_eval'])

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
        self._device = 'cuda' if self._use_cuda else 'cpu'
        if self._use_cuda:
            torch.cuda.set_device(self._rank)
            model.cuda()
        self._model = model
        self._enable_field = enable_field
        self._total_field = set(['learn', 'collect', 'eval', 'command'])
        if self._enable_field is None:
            self._init_learn()
            self._init_collect()
            self._init_eval()
            self._init_command()
        else:
            assert set(self._enable_field).issubset(self._total_field), self._enable_field
            for field in self._enable_field:
                getattr(self, '_init_' + field)()
        if self._use_distributed:
            if self._enable_field is None or self._enable_field == ['learn']:
                armor = self._armor
            else:
                armor = getattr(self, '_{}_armor'.format(self._enable_field[0]))
            for name, param in armor.model.state_dict().items():
                assert isinstance(param.data, torch.Tensor), type(param.data)
                broadcast(param.data, 0)
            for name, param in armor.model.named_parameters():
                setattr(param, 'grad', torch.zeros_like(param))

    def _create_model(self, cfg: dict, model: Optional[Union[type, torch.nn.Module]] = None) -> torch.nn.Module:
        model_cfg = cfg.model
        if model is None:
            if 'model_type' not in model_cfg:
                model_type, import_names = self.default_model()
                model_cfg.model_type = model_type
                model_cfg.import_names = import_names
            return create_model(model_cfg)
        else:
            if isinstance(model, type):
                return model(**model_cfg)
            elif isinstance(model, torch.nn.Module):
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

    @abstractmethod
    def _init_command(self) -> None:
        raise NotImplementedError

    @property
    def learn_mode(self) -> 'Policy.learn_function':  # noqa
        return Policy.learn_function(
            self._data_preprocess_learn, self._forward_learn, self._reset_learn, self.__repr__, self.state_dict_handle,
            self.set_setting, self._monitor_vars_learn, self._get_batch_size
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
            self.set_setting,
            self.state_dict_handle,
        )

    @property
    def eval_mode(self) -> 'Policy.eval_function':  # noqa
        return Policy.eval_function(
            self._data_preprocess_collect,
            self._forward_eval,
            self._data_postprocess_collect,
            self._reset_eval,
            self.set_setting,
            self.state_dict_handle,
        )

    @property
    def command_mode(self) -> 'Policy.command_function':  # noqa
        return Policy.command_function(self._get_setting_learn, self._get_setting_collect, self._get_setting_eval)

    def set_setting(self, mode_name: str, setting: dict) -> None:
        # this function is used in both collect and learn modes
        assert mode_name in ['learn', 'collect', 'eval'], mode_name
        for k, v in setting.items():
            # this attribute should be set in _init_{mode} method as a list
            assert k in getattr(self, '_' + mode_name + '_setting_set')
            setattr(self, '_' + k, v)

    def __repr__(self) -> str:
        return "nerveX DRL Policy\n{}".format(repr(self._model))

    def state_dict_handle(self) -> dict:
        state_dict = {'model': self._model}
        if hasattr(self, '_optimizer'):
            state_dict['optimizer'] = self._optimizer
        return state_dict

    def _monitor_vars_learn(self) -> List[str]:
        return ['cur_lr', 'total_loss']

    def sync_gradients(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                allreduce(param.grad.data)

    @abstractmethod
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

    def _get_batch_size(self) -> Union[int, Dict[str, int]]:
        return self._cfg.learn.batch_size

    # *************************************** collect function ************************************

    @abstractmethod
    def _data_preprocess_collect(self, data: Dict[int, Any]) -> Tuple[List[int], dict]:
        raise NotImplementedError

    @abstractmethod
    def _forward_collect(self, data_id: List[int], data: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _data_postprocess_collect(self, data_id: List[int], data: dict) -> Dict[int, dict]:
        raise NotImplementedError

    @abstractmethod
    def _process_transition(self, obs: Any, armor_output: dict, timestep: namedtuple) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _get_train_sample(self, traj_cache: deque) -> Union[None, List[Any]]:
        raise NotImplementedError

    @abstractmethod
    def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
        raise NotImplementedError

    # *************************************** eval function ************************************

    @abstractmethod
    def _forward_eval(self, data_id: List[int], data: dict) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
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


policy_mapping = {}


def create_policy(cfg: dict, **kwargs) -> Policy:
    cfg = EasyDict(cfg)
    import_module(cfg.import_names)
    if cfg.policy_type not in policy_mapping:
        raise KeyError("not support policy type: {}".format(cfg.policy_type))
    else:
        return policy_mapping[cfg.policy_type](cfg, **kwargs)


def register_policy(name: str, policy: type) -> None:
    assert issubclass(policy, Policy)
    assert isinstance(name, str)
    policy_mapping[name] = policy
