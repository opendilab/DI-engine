Env Overview
===================


Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

概述：
    环境模块从功能上有两种类别，一是通用的基类接口和公共函数，二是每一个具体问题环境的相关实现。

代码结构：
    主要分为如下几个子模块：

        1. env: 环境基类和具体的环境类（用于和智能体交互）, **唯一外部接口类和外部接口函数定义**
        2. common: 通用环境元素基类，公共函数（负责各种前后处理，与环境相关的数据转换，例：独热编码）
        3. 具体的问题环境实例(即一个问题用同一个文件夹管理)，实例中可能包括如下内容：

            1. observation: 具体环境观察类
            2. action: 具体环境动作类
            3. reward: 具体环境奖励类
            4. stat: 具体统计信息类（负责环境中各种统计信息的记录和使用，比如alphastar中的人类统计量z）
            5. config: 具体的配置文件
            6. other: 其他模块（一般存放不属于上述内容的模块，例如alphastar中的地图信息模块）
            7. tests: 单元测试

                示例如下：

                .. image:: alphastar_env_structure.png

基类定义：
    1. BaseEnv (env/base_env.py)

        .. code:: python

            class BaseEnv(ABC):
                """
                Overview: basic environment class
                Interface: __init__
                Property: timestep
                """
                timestep = namedtuple('BaseEnvTimestep', ['obs', 'act', 'reward', 'done', 'info'])
                info_template = namedtuple('BaseEnvInfo', ['agent_num', 'obs_space', 'act_space', 'rew_space'])

                @abstractmethod
                def __init__(self, cfg: dict) -> None:
                    raise NotImplementedError

                @abstractmethod
                def reset(self) -> Any:
                    raise NotImplementedError

                @abstractmethod
                def close(self) -> None:
                    raise NotImplementedError

                @abstractmethod
                def step(self, action: Any) -> 'BaseEnvTimestep':
                    raise NotImplementedError

                @abstractmethod
                def seed(self, seed: int) -> None:
                    raise NotImplementedError

                @abstractmethod
                def info(self) -> 'BaseEnv.info':
                    raise NotImplementedError

                @abstractmethod
                def __repr__(self) -> str:
                    raise NotImplementedError

                @abstractmethod
                def pack(self, timesteps: List['BaseEnvTimestep'] = None, obs: Any = None) -> 'BaseEnvTimestep':
                    raise NotImplementedError

                @abstractmethod
                def unpack(self, action: Any) -> List[Any]:
                    raise NotImplementedError

        - 概述：
            环境基类，用于和外部智能体进行交互

        - 类变量：
            1. timestep(namedtuple): 定义了环境每运行一步返回的时间步内容，一般包括'obs', 'act', 'reward', 'done', 'info'五部分，子类可以重写自己的该变量。
            2. info(namedtuple): 定义了环境的基本信息，例如环境中智能体的数量，观察空间的维度等等，可用于神经网络创建的输入参数等等，子类可以重写自己的该变量。


        - 类接口方法：
            1. __init__: 初始化
            2. reset: 重启环境(reset方法在子类的实现中可能会存在输入参数，比如一个episode结束重启时需要外部指定一些参数)
            3. close: 关闭环境，释放资源
            4. step: 环境执行输入的动作，完成一个时间步
            5. seed: 设置环境随机种子
            6. info: 返回环境基本信息，包含智能体数目，观察空间维度信息等
            7. __repr__: 返回环境类状态说明的字符串
            8. pack: (env->agent) 对多个环境的timestep或obs进行组装打包，外界可以按照原有的属性进行访问。例如timestep.reward就可获得所有环境的reward
            9. unpack: (agent->env) 对组装好的action进行拆分，拆解成可以直接传给各个环境的形式。

            .. note::

                具体问题的运行环境创建不应该在 `__init__` 方法中实现，因为存在创建模型实例但不运行的使用场景（比如获取环境observation的维度等信息），推荐在reset方法中\
                判断运行环境是否已创建，如果没有则进行创建再reset，如果有则直接reset已有环境。



    2. EnvElement (common/env_element.py)

        .. code:: python

            from nervex.utils import SingletonMetaclass


            class IEnvElement(ABC):
                @abstractmethod
                def __repr__(self) -> str:
                    raise NotImplementedError

                @property
                @abstractmethod
                def info(self) -> Any:
                    raise NotImplementedError


            class EnvElement(IEnvElement, metaclass=SingletonMetaclass):
                info_template = namedtuple('EnvElementInfo', ['shape', 'value', 'to_agent_processor', 'from_agent_processor'])
                _instance = None
                _name = 'EnvElement'

                def __init__(self, *args, **kwargs) -> None:
                    # placeholder
                    # self._shape = None
                    # self._value = None
                    # self._to_agent_processor = None
                    # self._from_agent_processor = None
                    self._init(*args, **kwargs)
                    self._check()

                @abstractmethod
                def _init(*args, **kwargs) -> None:
                    raise NotImplementedError

                def __repr__(self) -> str:
                    return '{}: {}'.format(self._name, self._details())

                @abstractmethod
                def _details(self) -> str:
                    raise NotImplementedError

                def _check(self) -> None:
                    flag = [
                        hasattr(self, '_shape'),
                        hasattr(self, '_value'),
                        hasattr(self, '_to_agent_processor'),
                        hasattr(self, '_from_agent_processor'),
                    ]
                    assert all(flag), 'this class {} is not a legal subclass of EnvElement({})'.format(self.__class__, flag)

                @property
                def info(self) -> 'EnvElementInfo':
                    return self.info_template(
                        shape=self._shape,
                        value=self._value,
                        to_agent_processor=self._to_agent_processor,
                        from_agent_processor=self._from_agent_processor
                    )



        - 概述：
            环境元素基类，observation，action，reward等可以视为环境元素，该类及其子类负责某一具体环境元素的基本信息和处理函数定义，均使用单例\
            模式设计，内部不维护任何状态变量，使得在系统中永远可以获得相同的元素实例，提供一致的信息和映射。该类及其子类是stateless的，维护静态
            的属性和方法。

        - 类变量：
            1. info_template: 环境元素信息模板，一般包括维度，取值情况，发送给智能体数据的处理函数，从智能体接收到数据的处理函数
            2. _instance: 实现单例模型所用的类变量，指向该类的唯一实例
            3. _name: 该类的唯一标识名

        - 类接口方法：
            1. __init__: 初始化，注意初始化完成后会调用 `_check` 方法检查是否合法
            2. info: 返回该元素类的基本信息和处理函数
            3. __repr__: 返回提供元素说明的字符串

        - 子类需继承重写方法：
            1. _init: 实际上的初始化方法，这样实现是为了让子类调用方法 `__init__` 时也必须调用 `_check` 方法，相当于 `__init__` 只是一层wrapper
            2. _check: 检查合法性方法，检查一个环境元素类是否实现了必需属性，子类可以拓展该方法，即重写该方法——调用父类的该方法+实现自身需要检查的部分
            3. _details: 元素类详细信息

    3. EnvElementRunner(common/env_element_runner.py)

        .. code:: python

            class IEnvElementRunner(IEnvElement):
                @abstractmethod
                def get(self, engine: BaseEnv) -> Any:
                    raise NotImplementedError

                @abstractmethod
                def reset(self, *args, **kwargs) -> None:
                    raise NotImplementedError


            class EnvElementRunner(IEnvElementRunner):
                def __init__(self, *args, **kwargs) -> None:
                    self._init(*args, **kwargs)
                    self._check()

                @abstractmethod
                def _init(self, *args, **kwargs) -> None:
                    # set self._core and other state variable
                    raise NotImplementedError

                def _check(self) -> None:
                    flag = [hasattr(self, '_core'), isinstance(self._core, EnvElement)]
                    assert all(flag), flag

                def __repr__(self) -> str:
                    return repr(self._core)

                @property
                def info(self) -> 'EnvElementInfo':
                    return self._core.info

        - 概述：
            环境元素运行时基类，使用装饰模式实现，负责运行时相关的状态管理（比如维护一些状态记录变量）和提供可能的多态机制（对静态处理函数返回的结果进行再加工）。
            在静态环境元素接口基础上，新增了 `get` 和 `reset` 接口。该类将对应的静态环境元素实例作为自己的一个成员变量 `_core` 进行管理。
        - 类变量：
            无
        - 类接口方法：
            1. info：来源于接口的父类，实际使用时调用静态元素的相应方法
            2. __repr__：来源于接口的父类，实际使用时调用静态元素的相应方法
            3. get：得到实际运行时的元素值，需要传入具体env对象，所有对env信息的访问集中在 `get` 方法中，建议访问信息通过env的property实现
            4. reset：重启状态，一般需要在env重启时对应进行调用
        - 子类需继承重写方法：
            1. _init: 实际上的初始化方法，这样实现是为了让子类调用方法 `__init__` 时也必须调用 `_check` 方法，相当于 `__init__` 只是一层wrapper
            2. _check: 检查合法性方法，检查一个环境元素类是否实现了必需属性，子类可以拓展该方法，即重写该方法——调用父类的该方法+实现自身需要检查的部分

    .. note::


        1. `EnvElement` 和 `EnvElementRunner` 两个类构成完整的环境元素，其中前者代表静态不变的信息(stateless)，后者负责运行时变化的信息(stateful)，建议与特定环境元素相关的状态变量一律放在这里维护，env中只维护通用的状态变量
        2. 环境元素部分简易的类逻辑图如下：

            .. image:: env_element_class.png

.. note::

    1. 所有代码实现中命名一律使用名词单数，约定为习惯
    2. 所有代码实现秉承 **自身对外界输入质疑，自身对外界输出负责** 的思想，对输入参数做必要的check，对输出（返回值）明确规定其格式
    3. 环境元素的键值如果为空时，一律使用 `None`, 从重构版本开始废除 `'none'` 的用法。
