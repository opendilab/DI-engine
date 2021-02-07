Armor Overview
===================


Armor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

概述：
    Armor, 即智能体，该模块的设计初衷是维护模型运行时的信息和状态，Model和Armor共同组成了运行时的策略，其中前者定义了模型（神经网络）的计算图，后者则维护运行时变量。

代码结构：
    主要分为如下几个子模块：

        1. base_armor: 基础的armor定义，维护一些通用的接口方法
        2. armor_plugin: 每个问题相关的armor实例可以注册自己的插件(plugin)，比如RNN-based模型需要在运行时维护隐状态(hidden-state)。注意具体的插件类型是由实例创建时的plugin_cfg决定，所以同一个armor类的不同实例也可以加载不同插件（比如支持同一个armor在训练和测试时的不同需求）


基类定义：
    1. BaseArmor (worker/armor/base_armor.py)

        .. code:: python

            class BaseArmor(ABC):
                def __init__(self, model: torch.nn.Module, plugin_cfg: Union[OrderedDict, None]) -> None:
                    self._model = model
                    register_plugin(self, plugin_cfg)

                def forward(self, data: Any, param: Optional[dict] = None) -> Any:
                    if param is not None:
                        return self._model(data, param)
                    else:
                        return self._model(data)

                def mode(self, train: bool) -> None:
                    if train:
                        self._model.train()
                    else:
                        self._model.eval()

                @property
                def model(self) -> torch.nn.Module:
                    return self._model

                def state_dict(self) -> dict:
                    return {'model': self._model.state_dict()}

                def load_state_dict(self, state_dict: dict) -> None:
                    self._model.load_state_dict(state_dict['model'])

                def reset(self) -> None:
                    pass

        - 概述：
            智能体(armor)基类，和模型(model)组合构成运行时的智能体。该基类只提供通用的接口方法。

        - 类接口方法：
            1. __init__: 初始化。模型的创建应该在外部调用者处完成，作为参数传入，插件配置(plugin_cfg要么为None，要么是一个 **有序** 字典)
            2. forward: 智能体执行前向计算图。注意对于模型所需的超参数，统一放在param项中传入，模型内部自己进行解析，若无参数则将param置为None。而对于 ``forward`` 时是否计算梯度，可通过梯度插件进行管理。
            3. mode: 该方法是对 `torch.nn.Module` 的 `train/eval` 方法的封装，具体表现可以参见 `传送门 <https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.eval>`_
            4. model: 该property返回模型
            5. state_dict: 返回当前的状态信息(state_dict)，默认只返回模型的状态信息，子类可以重写该方法在字典中加入其它需要返回的信息。
            6. load_state_dict: 加载状态信息，子类也可进行重写
            7. reset: 重置智能体相关状态


    2. IArmorPlugin (worker/armor/armor_plugin.py)

        .. code:: python

            class IArmorPlugin(ABC):
                @abstractclassmethod
                def register(cls: type, armor: Any, **kwargs) -> None:
                    """inplace modify armor"""
                    raise NotImplementedError


            IArmorStatelessPlugin = IArmorPlugin


            class IArmorStatefulPlugin(IArmorPlugin):
                @abstractmethod
                def __init__(self, *args, **kwargs) -> None:
                    raise NotImplementedError


        - 概述：

            智能体插件分为两类，有状态(stateful)和无状态(stateless)插件，区别在于前者需要创建具体实例来维护相关信息，这个新创建的插件实例也会绑定到原来的armor实例上，作为其某个成员变量。
            两种插件都是对armor进行原地操作，即通过类方法 ``register`` 对输入的armor进行原地修改。

        - 目前已经实现的插件：

          1. 梯度插件(stateless)：控制 ``forward`` 时是否需要为计算梯度做准备（例如缓存中间计算结果）
          2. 隐状态插件(stateful): 控制 ``forward`` 时隐状态的行为，在实例内部根据训练batch样本数维护对应的隐状态，每次 ``forward`` 前输入上一次迭代的输出隐状态，而 ``forward`` 后保存该次的输出隐状态为下一次做准备，此外，该插件支持的特殊行为有：

                1. 单次迭代进输入部分样本，使用其对应的隐状态
                2. 对具体的样本的隐状态进行重置。

.. note::
    BaseArmor和Armor相关插件的测试可以参见 `worker/armor/tests/test_armor.py`
