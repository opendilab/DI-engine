from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Optional, List, Dict, Any, Tuple, Union

import torch
import copy
from easydict import EasyDict

from ding.model import create_model
from ding.utils import import_module, allreduce, broadcast, get_rank, allreduce_async, synchronize, deep_merge_dicts, \
    POLICY_REGISTRY


class Policy(ABC):
    """
    Overview:
        The basic class of Reinforcement Learning (RL) and Imitation Learning (IL) policy in DI-engine.
    Property:
        ``cfg``, ``learn_mode``, ``collect_mode``, ``eval_mode``
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        """
        Overview:
            Get the default config of policy. This method is used to create the default config of policy.
        Returns:
            cfg (:obj:`EasyDict`): The default config of corresponding policy. For the derived policy class, \
                it will recursively merge the default config of base class and its own default config.

        .. tip::
            This method will deepcopy the ``config`` attribute of the class and return the result. So users don't need \
            to worry about the modification of the returned config.
        """
        if cls == Policy:
            raise RuntimeError("Basic class Policy doesn't have completed default_config")

        base_cls = cls.__base__
        if base_cls == Policy:
            base_policy_cfg = EasyDict(copy.deepcopy(Policy.config))
        else:
            base_policy_cfg = copy.deepcopy(base_cls.default_config())
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg = deep_merge_dicts(base_policy_cfg, cfg)
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
    config = dict(
        on_policy=False,
        cuda=False,
        multi_gpu=False,
        bp_update_sync=True,
        traj_len_inf=False,
        model=dict(),
    )

    def __init__(
            self,
            cfg: EasyDict,
            model: Optional[torch.nn.Module] = None,
            enable_field: Optional[List[str]] = None
    ) -> None:
        """
        Overview:
            Initialize policy instance according to input configures and model. This method will initialize differnent \
            fields in policy, including ``learn``, ``collect``, ``eval``. The ``learn`` field is used to train the \
            policy, the ``collect`` field is used to collect data for training, and the ``eval`` field is used to \
            evaluate the policy. The ``enable_field`` is used to specify which field to initialize, if it is None, \
            then all fields will be initialized.
        Arguments:
            - cfg (:obj:`EasyDict`): The final merged config used to initialize policy. For the default config, \
                see the ``config`` attribute and its comments of policy class.
            - model (:obj:`torch.nn.Module`): The neural network model used to initialize policy. If it \
                is None, then the model will be created according to ``default_model`` method and ``cfg.model`` field. \
                Otherwise, the model will be set to the ``model`` instance created by outside caller.
            - enable_field (:obj:`Optional[List[str]]`): The field list to initialize. If it is None, then all fields \
                will be initialized. Otherwise, only the fields in ``enable_field`` will be initialized, which is \
                beneficial to save resources.

        .. note::
            For the derived policy class, it should implement the ``_init_learn``, ``_init_collect``, ``_init_eval`` \
            method to initialize the corresponding field.
        """
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
                multi_gpu = self._cfg.multi_gpu
                self._rank = get_rank() if multi_gpu else 0
                if self._cuda:
                    model.cuda()
                if multi_gpu:
                    bp_update_sync = self._cfg.bp_update_sync
                    self._bp_update_sync = bp_update_sync
                    self._init_multi_gpu_setting(model, bp_update_sync)
            else:
                self._rank = 0
                if self._cuda:
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
        """
        Overview:
            Initialize multi-gpu data parallel training setting, including broadcast model parameters at the beginning \
            of the training, and prepare the hook function to allreduce the gradients of model parameters.
        Arguments:
            - model (:obj:`torch.nn.Module`): The neural network model to be trained.
            - bp_update_sync (:obj:`bool`): Whether to synchronize update the model parameters after allreduce the \
                gradients of model parameters. Async update can be parallel in different network layers like pipeline \
                so that it can save time.
        """
        for name, param in model.state_dict().items():
            assert isinstance(param.data, torch.Tensor), type(param.data)
            broadcast(param.data, 0)
        # here we manually set the gradient to zero tensor at the beginning of the training, which is necessary for
        # the case that different GPUs have different computation graph.
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

    def _create_model(self, cfg: EasyDict, model: Optional[torch.nn.Module] = None) -> torch.nn.Module:
        """
        Overview:
            Create neural network model according to input configures and model. If the input model is None, then \
            the model will be created according to ``default_model`` method and ``cfg.model`` field. Otherwise, the \
            model will be set to the ``model`` instance created by outside caller.
        Arguments:
            - cfg (:obj:`EasyDict`): The final merged config used to initialize policy.
            - model (:obj:`torch.nn.Module`): The neural network model used to initialize policy. User can refer to \
                the default model defined in corresponding policy to customize its own model.
        Returns:
            - model (:obj:`torch.nn.Module`): The created neural network model. Then different modes of policy will \
                add wrappers and plugins to the model, which is used to train, collect and evaluate.
        Raises:
            - RuntimeError: If the input model is not None and is not an instance of ``torch.nn.Module``.
        """
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
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. This method will be \
            called in ``__init__`` method if ``learn`` field is in ``enable_field``. Almost different policies have \
            its own learn mode, so this method must be overrided in subclass.

        .. note::
            For the member variables that need to be saved and loaded, please refer to the ``_state_dict_learn`` \
            and ``_load_state_dict_learn`` methods.

        .. note::
            For the member variables that need to be monitored, please refer to the ``_monitor_vars_learn`` method.

        .. note::
            If you want to set some spacial member variables in ``_init_learn`` method, you'd better name them \
            with prefix ``_learn_`` to avoid conflict with other modes, such as ``self._learn_attr1``.
        """
        raise NotImplementedError

    @abstractmethod
    def _init_collect(self) -> None:
        """
        Overview:
            Initialize the collect mode of policy, including related attributes and modules. This method will be \
            called in ``__init__`` method if ``collect`` field is in ``enable_field``. Almost different policies have \
            its own collect mode, so this method must be overrided in subclass.

        .. note::
            For the member variables that need to be saved and loaded, please refer to the ``_state_dict_collect`` \
            and ``_load_state_dict_collect`` methods.

        .. note::
            If you want to set some spacial member variables in ``_init_collect`` method, you'd better name them \
            with prefix ``_collect_`` to avoid conflict with other modes, such as ``self._collect_attr1``.
        """
        raise NotImplementedError

    @abstractmethod
    def _init_eval(self) -> None:
        """
        Overview:
            Initialize the eval mode of policy, including related attributes and modules. This method will be \
            called in ``__init__`` method if ``eval`` field is in ``enable_field``. Almost different policies have \
            its own eval mode, so this method must be overrided in subclass.

        .. note::
            For the member variables that need to be saved and loaded, please refer to the ``_state_dict_eval`` \
            and ``_load_state_dict_eval`` methods.

        .. note::
            If you want to set some spacial member variables in ``_init_eval`` method, you'd better name them \
            with prefix ``_eval_`` to avoid conflict with other modes, such as ``self._eval_attr1``.
        """
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
        """
        Overview:
            In order to control the access of the policy attributes, we expose different modes to outside rather than \
            directly use the policy instance. And we also provide a method to set the attribute of the policy in \
            different modes. And the new attribute will named as ``_{name}``.
        Arguments:
            - name (:obj:`str`): The name of the attribute.
            - value (:obj:`Any`): The value of the attribute.
        """
        setattr(self, '_' + name, value)

    def _get_attribute(self, name: str) -> Any:
        """
        Overview:
            In order to control the access of the policy attributes, we expose different modes to outside rather than \
            directly use the policy instance. And we also provide a method to get the attribute of the policy in \
            different modes.
        Arguments:
            - name (:obj:`str`): The name of the attribute.
        Returns:
            - value (:obj:`Any`): The value of the attribute.

        .. note::
            DI-engine's policy will first try to access `_get_{name}` method, and then try to access `_{name}` \
            attribute. If both of them are not found, it will raise a ``NotImplementedError``.
        """
        if hasattr(self, '_get_' + name):
            return getattr(self, '_get_' + name)()
        elif hasattr(self, '_' + name):
            return getattr(self, '_' + name)
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        """
        Overview:
            Get the string representation of the policy.
        Returns:
            - repr (:obj:`str`): The string representation of the policy.
        """
        return "DI-engine DRL Policy\n{}".format(repr(self._model))

    def sync_gradients(self, model: torch.nn.Module) -> None:
        """
        Overview:
            Synchronize (allreduce) gradients of model parameters in data-parallel multi-gpu training.
        Arguments:
            - model (:obj:`torch.nn.Module`): The model to synchronize gradients.

        .. note::
            This method is only used in multi-gpu training, and it shoule be called after ``backward`` method and \
            before ``step`` method. The user can also use ``bp_update_sync`` config to control whether to synchronize \
            gradients allreduce and optimizer updates.
        """

        if self._bp_update_sync:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    allreduce(param.grad.data)
        else:
            synchronize()

    # don't need to implement default_model method by force
    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default neural network model setting for demonstration. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): The registered model name and model's import_names.

        .. note::
            The user can define and use customized network model but must obey the same inferface definition indicated \
            by import_names path. For example about DQN, its registered name is ``dqn`` and the import_names is \
            ``ding.model.template.q_learning.DQN``
        """
        raise NotImplementedError

    # *************************************** learn function ************************************

    @abstractmethod
    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        raise NotImplementedError

    # don't need to implement _reset_learn method by force
    def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
        pass

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.

        .. tip::
            The default implementation is ``['cur_lr', 'total_loss']``. Other derived classes can overwrite this \
            method to add their own keys if necessary.
        """
        return ['cur_lr', 'total_loss']

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): The dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _get_batch_size(self) -> Union[int, Dict[str, int]]:
        # some specifial algorithms use different batch size for different optimization parts.
        if 'batch_size' in self._cfg:
            return self._cfg.batch_size
        else:  # for compatibility
            return self._cfg.learn.batch_size

    # *************************************** collect function ************************************

    @abstractmethod
    def _forward_collect(self, data: dict, **kwargs) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _process_transition(
            self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]], policy_output: Dict[str, torch.Tensor],
            timestep: namedtuple
    ) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Process and pack one timestep transition data into a dict, such as <s, a, r, s', done>. Some policies \
            need to do some special process and pack its own necessary attributes (e.g. hidden state and logit), \
            so this method is left to be implemented by the subclass.
        Arguments:
            - obs (:obj:`Union[torch.Tensor, Dict[str, torch.Tensor]]`): The observation of the current timestep.
            - policy_output (:obj:`Dict[str, torch.Tensor]`): The output of the policy network with the observation \
                as input. Usually, it contains the action and the logit of the action.
            - timestep (:obj:`namedtuple`): The execution result namedtuple returned by the environment step method, \
                except all the elements have been transformed into tensor data. Usually, it contains the next obs, \
                reward, done, info, etc.
        Returns:
            - transition (:obj:`Dict[str, torch.Tensor]`): The processed transition data of the current timestep.
        """
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

    def _get_n_sample(self):
        if 'n_sample' in self._cfg:
            return self._cfg.n_sample
        else:  # for compatibility
            return self._cfg.collect.get('n_sample', None)  # for some adpative collecting data case

    def _get_n_episode(self):
        if 'n_episode' in self._cfg:
            return self._cfg.n_episode
        else:  # for compatibility
            return self._cfg.collect.get('n_episode', None)  # for some adpative collecting data case

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
    """
    Overview:
        Policy with command mode, which can be used in old version of DI-engine pipeline: ``serial_pipeline``. \
        ``CommandModePolicy`` uses ``_get_setting_learn``, ``_get_setting_collect``, ``_get_setting_eval`` methods \
        to exchange information between different workers.

    Interface:
        ``_init_command``, ``_get_setting_learn``, ``_get_setting_collect``, ``_get_setting_eval``
    Property:
        ``command_mode``
    """
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


def create_policy(cfg: EasyDict, **kwargs) -> Policy:
    """
    Overview:
        Create a policy instance according to ``cfg`` and other kwargs.
    Arguments:
        - cfg (:obj:`EasyDict`): Final merged policy config.
    ArgumentsKeys:
        - type (:obj:`str`): Policy type set in ``POLICY_REGISTRY.register`` method , such as ``dqn`` .
        - import_names (:obj:`List[str]`): A list of module names (paths) to import before creating policy, such \
            as ``ding.policy.dqn`` .
    Returns:
        - policy (:obj:`Policy`): The created policy instance.

    .. tip::
        ``kwargs`` contains other arguments that need to be passed to the policy constructor. You can refer to \
        the ``__init__`` method of the corresponding policy class for details.

    .. note::
        For more details about how to merge config, please refer to the system document of DI-engine \
        (`en link <../03_system/config.html>`_).
    """
    import_module(cfg.get('import_names', []))
    return POLICY_REGISTRY.build(cfg.type, cfg=cfg, **kwargs)


def get_policy_cls(cfg: EasyDict) -> type:
    """
    Overview:
        Get policy class according to ``cfg``, which is used to access related class variables/methods.
    Arguments:
        - cfg (:obj:`EasyDict`): Final merged policy config.
    ArgumentsKeys:
        - type (:obj:`str`): Policy type set in ``POLICY_REGISTRY.register`` method , such as ``dqn`` .
        - import_names (:obj:`List[str]`): A list of module names (paths) to import before creating policy, such \
            as ``ding.policy.dqn`` .
    Returns:
        - policy (:obj:`type`): The policy class.
    """
    import_module(cfg.get('import_names', []))
    return POLICY_REGISTRY.get(cfg.type)
