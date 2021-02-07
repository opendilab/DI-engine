Policy Overview
===================


Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

概述：
    算法策略模块是所实现的强化学习算法的核心内容，对应的功能包括：控制算法训练过程、算法数据的部分预处理、算法使用的神经网络的建立等等。
    Policy包含了以下四个子mode：
    
    1. ``learn_mode``: learner执行过程中需要 ``Policy`` 定义的函数，如learner step的 ``forward`` 函数即为 ``learn_mode.forward`` ，在learner中的调用为:


        .. code:: python

            data = self._policy.data_preprocess(data)
            log_vars = self._policy.forward(data)。


    2. ``collect_mode``: actor执行过程中需要 ``Policy`` 定义的函数，如actor step的forward函数即为 ``collect_mode.forward`` ， trainsition的处理为 ``collect_mode.process_transition`` , 在actor中的调用为:

        .. code:: python

            transition = self._policy.process_transition(
                self._obs_pool[env_id], self._policy_output_pool[env_id], timestep
            )

    3. ``eval_mode``: evaluator执行过程中需要 ``Policy`` 定义的函数，如evaluator  step的forward函数即为 ``collect_mode.forward`` 。
    4. ``command_mode``: 我们通过command来控制一些参数，比如command中实现了eps-greedy控制信息从learn step到具体collect eps使用值的信息传递：

        .. code:: python
            
            class BaseSerialCommand(object):

                def step(self) -> None:
                    # update info
                    learner_info = self._learner.learn_info # here info contain learn step
                    self._info.update(learner_info)
                    # update setting
                    collect_setting = self._policy.get_setting_collect(self._info) # here  get collect eps
                    # set setting
                    self._actor.policy.set_setting('collect', collect_setting) # here  set colllect eps
            

代码结构：
    主要分为以下两种：

        1. Policy基类: Policy模块的基类。
            - 所有Policy都需要继承``base_policy``中的``Policy``类；
            - 由于很多算法 ``Policy`` 在控制流程上有共同点，为了方便我们提取了 ``common_policy`` 中的 ``CommonPolicy`` 类，该类继承了 ``Policy`` 类； ``CommonPolicy`` 类不是必须继承的，如有具体需求可以用户自己写。
            - Policy在定义后，需要使用调用 ``base_policy`` 中的 ``register_policy进行`` 注册;使用时需要根据对应的config文件使用 ``create_policy`` 创建实例； ``regist_policy`` 的使用可见 `tutorial部分 <../tutorial/index.html>`_ 。 ``create_policy`` 则是在算法执行时，根据config文件中算法策略和config参数在执行过程中调用。


        2. Policy具体实现: 我们有很多已实现的算法Policy，位于nervex/policy文件夹下。


基类定义：
    1. ``Policy`` (policy/base_policy.py)

        .. code:: python

            class Policy(ABC):
                learn_function = namedtuple(
                    'learn_function',
                    ['data_preprocess', 'forward', 'reset', 'info', 'state_dict_handle', 'set_setting', 'monitor_vars']
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
                        self._data_preprocess_learn,
                        self._forward_learn,
                        self._reset_learn,
                        self.__repr__,
                        self.state_dict_handle,
                        self.set_setting,
                        self._monitor_vars_learn,
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
                def _data_preprocess_learn(self, data: List[Any]) -> dict:
                    raise NotImplementedError

                @abstractmethod
                def _forward_learn(self, data: dict) -> Dict[str, Any]:
                    raise NotImplementedError

                @abstractmethod
                def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
                    raise NotImplementedError

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


        - 概述：
            策略基类，算法策略与运行入口（entry，包括串并行）进行交互，定义了相关的接口和调用方式。

        - 类变量：
            1. ``learn_function`` (namedtuple): 定义了策略在learning过程中需要被调用的函数/方法，包括'data_preprocess', 'forward', 'reset', 'info', 'state_dict_handle', 'set_setting', 'monitor_vars'。
            2. ``collect_function`` (namedtuple): 定义了策略在收集actor数据过程中需要被调用的函数/方法，包括'data_preprocess', 'forward', 'data_postprocess', 'process_transition', 'get_train_sample', 'reset','set_setting', 'state_dict_handle'。
            3. ``eval_function`` (namedtuple): 定义了策略在eval当前算法效果时过程中需要被调用的函数/方法，包括'data_preprocess', 'forward', 'data_postprocess', 'reset', 'set_setting', 'state_dict_handle'。

            .. note::

                三个类变量与learn_mode, collect_mode, eval_mode返回相对应。

        - 接口方法：
            1. ``__init__``: 通过对应传入的config文件初始化，支持在初始化时传入该实例使用的模型以及控制可调用哪些函数
            2. ``learn_mode``: learning过程中需要使用到的函数，包括
                - ``data_preprocess``：由 ``self._data_preprocess_learn`` 实现，在数据传入forward前进行预处理。
                - ``forward``： 由 ``self._forward_learn`` 实现，包括算法loss的计算，梯度下降优化模型的过程等等。
                - ``reset``： 由 ``self._reset_learn`` 实现，通常包括模型状态的reset，和模型是否需要梯度的设置等(is_train = True)。
                - ``info``： 由 ``self.__repr__`` 实现，策略名的描述。
                - ``state_dict_handle``： 由 ``self.state_dict_handle`` 实现，返回当前模型及优化器的参数。
                - ``set_setting``：由 ``self.set_setting`` 实现，设置learn和collect中需要用到的相关参数。
                - ``monitor_vars``： 由 ``self._monitor_vars_learn`` 实现，设置logger需要监控的相关数据，如当前学习率，loss等等。
            3. ``collect_mode``:
                - ``data_preprocess``：由 ``self._data_preprocess_collect`` 实现，在数据传入forward前进行预处理。
                - ``forward``： 由 ``self._forward_collect`` 实现，根据对应的输入采集action，如epsilon greedy的参数需要传入。
                - ``reset``：由``self._reset_collec`` 实现，通常包括模型状态的reset，和模型是否需要梯度的设置等(is_train = False)。
                - ``data_postprocess``：由 ``self._data_postprocess_collect`` 实现，处理forward之后传入buffer之前的数据。
                - ``process_transition``：由 ``self._process_transition`` 实现, 根据obs、armor的输出，环境的timestep处理得到数据帧。
                - ``get_train_sample``：由 ``self._get_train_sample`` 实现, 从trajectory中选取合适的数据所为训练样本，通常使用adder。
                - ``set_setting``：由 ``self.set_setting`` 实现，设置learn和collect中需要用到的相关参数，如epsilon greedy的参数等。
                - ``state_dict_handle``： 由 ``self.state_dict_handle`` 实现，返回当前模型及优化器的参数。
            4. ``eval_mode``:
                - ``data_preprocess``：由 ``self._data_preprocess_collect`` 实现，在数据传入forward前进行预处理，与collect相同。
                - ``forward``： 由 ``self._forward_eval`` 实现，根据对应的输入采集action。
                - ``reset``：由 ``self._reset_eval`` 实现，通常包括模型状态的reset，和模型是否需要梯度的设置等(is_train = False)。
                - ``data_postprocess``：由 ``self._data_postprocess_collect`` 实现，处理forwrad之后传入buffer之前的数据，与collect相同。
                - ``process_transition``：由 ``self._process_transition`` 实现, 根据obs、armor的输出，环境的timestep处理得到数据帧。
                - ``set_setting``：由 ``self.set_setting`` 实现，设置learn和collect中需要用到的相关参数。
                - ``state_dict_handle``： 由 ``self.state_dict_handle`` 实现，返回当前模型及优化器的参数。 
            5. ``command_mode``: 对应的learn，collect，eval的相关setting参数
                - 包括 ``self._get_setting_learn`` ， ``self._get_setting_collect`` ， ``self._get_setting_eval`` , 以字典格式返回参数。
            6. ``default_model``: 没有设置model时，算法策略默认采用的model。
            7. ``sync_gradients``: 对于分布式神经网络模型的数据并行训练，反向传播之后优化器更新之前，需要调用此函数同步梯度。
        
        - 子类需继承重写方法：
            该子类需要重写``Policy``内所有未被实现的方法，即接口方法中被提及的所有实现。


        .. note::
            以上是``Policy``的一些基本的介绍和使用情况，具体写法可以参考我们位于nervex/policy文件夹下已实现的一些算法。
            
        .. note::
            ``transition`` 即是在actor执行过程中，环境每一次step后留下的相应记录（至少应包括'obs', 'action', 'reward', 'done'等)。
            ``transition`` 随后会通过 ``get_train_sample`` 后，变成learner处训练所需要的格式，加入到buffer中。

        .. tip::
            了解``Policy``是如何被具体使用，请参考我们的入口文件或者我们的 `tutorial部分 <../tutorial/index.html>`_ 。
            

        .. warning::

            算法的``model``可能包括本身的model和target model，在实现策略时时请不要忘记对``target model``进行相关处理!
                

    2. ``CommonPolicy`` (policy/common_policy.py)

        .. code:: python

            from .base_policy import Policy


            class CommonPolicy(Policy):

                def _data_preprocess_learn(self, data: List[Any]) -> dict:
                    # data preprocess
                    data = default_collate(data)
                    ignore_done = self._cfg.learn.get('ignore_done', False)
                    if ignore_done:
                        data['done'] = None
                    else:
                        data['done'] = data['done'].float()
                    use_priority = self._cfg.get('use_priority', False)
                    if use_priority:
                        data['weight'] = data['IS']
                    else:
                        data['weight'] = data.get('weight', None)
                    if self._use_cuda:
                        data = to_device(data, 'cuda:{}'.format(self._rank % 8))
                    return data

                def _data_preprocess_collect(self, data: Dict[int, Any]) -> Tuple[List[int], dict]:
                    data_id = list(data.keys())
                    data = default_collate(list(data.values()))
                    if self._use_cuda:
                        data = to_device(data, 'cuda')
                    data = {'obs': data}
                    return data_id, data

                def _data_postprocess_collect(self, data_id: List[int], data: dict) -> Dict[int, dict]:
                    if self._use_cuda:
                        data = to_device(data, 'cpu')
                    data = default_decollate(data)
                    return {i: d for i, d in zip(data_id, data)}

                def _get_train_sample(self, traj_cache: deque) -> Union[None, List[Any]]:
                    # adder is defined in _init_collect
                    data = self._adder.get_traj(traj_cache, self._traj_len)
                    return self._adder.get_train_sample(data)

                def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
                    self._armor.mode(train=True)
                    self._armor.reset(data_id=data_id)

                def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
                    self._collect_armor.mode(train=False)
                    self._collect_armor.reset(data_id=data_id)

                def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
                    self._eval_armor.mode(train=False)
                    self._eval_armor.reset(data_id=data_id)

                def _get_setting_learn(self, *args, **kwargs) -> dict:
                    return {}

                def _get_setting_collect(self, *args, **kwargs) -> dict:
                    return {}

                def _get_setting_eval(self, *args, **kwargs) -> dict:
                    return {}



        - 概述：
            一些 ``Policy`` 中方法的简单实现，使用 ``CommonPolicy`` 可以减少重复代码量。

        - 类接口方法：
            1. ``_data_preprocess_learn``: 使用default_collect, 处理'weight', 'done'等参数，并将数据转换到模型所在的设备
            2. ``_data_preprocess_collect``: 使用default_collect, 得到相应'obs'，并将数据转换到模型所在的设备
            3. ``_data_postprocess_collect``: 使用default_collect, 并将数据转换到cpu
            4. ``_get_train_sample``: 使用adder，详见adder_overview
            5. ``_reset_learn``: reset learner的model， 设置train=True
            6. ``_reset_collect``: reset actor的model， 设置train=False
            7. ``_reset_eval``: reset evaluator的model， 设置train=False
            8. ``_get_setting_learn``: 返回空dict
            9. ``_get_setting_collect``: 返回空dict
            10. ``_get_setting_eval``: 返回空dict

        - 子类需继承重写方法：
            该子类需要重写``Policy``中接口方法被提及的所有未被``CommonPolicy``实现的方法

    3. ``policy_mapping`` 及 ``Policy`` 的使用(policy/base_policy.py)

        .. code:: python

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

        - 概述：
            我们通过 ``policy_mapping`` 的方式存储和调用我们实现的各种算法的 ``Policy`` 类，``register_policy`` 将 ``Policy`` 类存入 ``policy_mapping`` ， ``create_policy`` 根据 ``policy_mapping`` 里的 ``Policy`` 类创造实例。
        - 变量：
           ``policy_mapping``
        
    .. tip::

        写完 ``Policy`` 类后不要忘记 ``register_policy`` 。
