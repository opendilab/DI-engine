Env Overview
===================


Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

概述：
    环境模块从功能上由两部分组成，一是通用的环境基类接口，二是环境中的元素基类和公共函数。具体的环境实例可以参见 ``app_zoo`` 中相关的环境。

代码结构：
    主要分为如下几个子模块：

        1. env: 环境基类和具体的环境类（用于和智能体交互）, **唯一外部接口类和外部接口函数定义**
        2. common: 通用环境元素基类，公共函数（负责各种前后处理，与环境相关的数据转换，例：独热编码）

    .. note::

        对于具体的问题环境实例，一个环境用同一个文件夹管理，实例中可能包括如下内容：

            1. observation: 具体环境观察类
            2. action: 具体环境动作类
            3. reward: 具体环境奖励类
            4. config: 具体的配置文件
            5. other: 其他模块（一般存放不属于上述内容的模块，例如alphastar中的地图信息模块）
            6. tests: 单元测试

                示例如下：

                .. image:: alphastar_env_structure.png

基类定义：
    1. BaseEnv (env/base_env.py)

        .. code:: python

            BaseEnvTimestep = namedtuple('BaseEnvTimestep', ['obs', 'reward', 'done', 'info'])
            BaseEnvInfo = namedtuple('BaseEnvInfo', ['agent_num', 'obs_space', 'act_space', 'rew_space'])


            class BaseEnv(ABC):

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
                def step(self, action: Any) -> 'BaseEnv.timestep':
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

                @staticmethod
                def create_collector_env_cfg(cfg: dict) -> List[dict]:
                    collector_env_num = cfg.pop('collector_env_num', 1)
                    return [cfg for _ in range(collector_env_num)]

                @staticmethod
                def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
                    evaluator_env_num = cfg.pop('evaluator_env_num', 1)
                    return [cfg for _ in range(evaluator_env_num)]

                # optional method
                def enable_save_replay(self, replay_path: str) -> None:
                    raise NotImplementedError


        - 概述：
            环境基类，用于和外部策略进行交互

        - 全局变量（也可定义为类变量）：
            1. BaseEnvTimestep(namedtuple): 定义了环境每运行一步返回的内容，一般包括'obs', 'act', 'reward', 'done', 'info'五部分，子类可以自定义自己的该变量，但注意必须包含上述五个字段。
            2. BaseEnvInfo(namedtuple): 定义了环境的基本信息，例如环境中智能体的数量，观察空间的维度等等，可用于神经网络创建的输入参数等等，一般包括'agent_num', 'obs_space', 'act_space', 'rew_space'四部分，其中 ``xxx_space`` 必须使用 ``envs/common/env_element.py`` 中的 ``EnvElementInfo`` 进行创建，子类可以自定义自己的该变量，为其增加新的字段。

            .. note::

                此外， ``obs_space`` 和 ``subprocess_env_manager`` 中 ``shared_memory`` 的相关使用存在强依赖，如要使用则必须按照 ``EnvElementInfo`` 来实现。


        - 接口方法：
            1. __init__: 初始化
            2. reset: 重启环境(reset方法在子类的实现中可能会存在输入参数，比如一个episode结束重启时需要外部指定一些参数)
            3. close: 关闭环境，释放资源
            4. step: 环境执行输入的动作，完成一个时间步
            5. seed: 设置环境随机种子
            6. info: 返回环境基本信息，包含智能体数目，观察空间维度信息等
            7. __repr__: 返回环境类状态说明的字符串
            8. create_collector_env_cfg: 为数据收集创建相应的环境配置文件，与 ``create_evaluator_env_cfg`` 互相独立，便于使用者对数据收集和性能评测设置不同的环境参数，根据传入的初始配置为每个具体的环境生成相应的配置文件，默认情况会获取配置文件中的环境个数，然后将默认环境配置复制相应份数返回
            9. create_evaluator_env_cfg: 为性能评测创建相应的环境配置文件，功能同上说明
            10. enable_save_replay: 使环境可以保存运行过程为视频文件，便于调试和可视化，一般在环境开始实际运行前调用，功能上代替常见环境中的render方法。（该方法可选实现）

            .. note::

                对于一个环境的具体创建（例如打开其他模拟器客户端），该行为不应该在 ``__init__`` 方法中实现，因为存在创建模型实例但不运行的使用场景（比如获取环境observation的维度等信息），推荐在 ``reset`` 方法中实现，即判断运行环境是否已创建，如果没有则进行创建再reset，如果有则直接reset已有环境。如果使用者依然想要在 ``__init__`` 方法中完成该功能，请自行确认不会有资源浪费或冲突的情况发生。

            .. note::

                关于BaseEnvInfo和BaseEnvTimestep，如无特殊需求可以直接调用nervex提供的默认定义，即：

                .. code:: python

                    from nervex.envs import BaseEnvTimestep, BaseEnvInfo

                如果需要自定义，按照上文的要求使用 ``namedtuple`` 实现即可。

            .. tip::

                ``seed`` 方法的调用一般在 ``__init__`` 方法之后，``reset`` 方法之前。如果将模型的创建放在 ``reset`` 方法中，则 ``seed`` 方法只需要记录下这个值，在 ``reset`` 方法执行时设置随机种子即可。

            .. warning::

                nervex对于环境返回的 ``info`` 字段有一些依赖关系, ``info`` 是一个dict，其中某些键值对会有相关依赖要求：
                
                1. `final_eval_reward`: 环境一个episode结束时（done=True）必须包含该键值，值为float类型，表示环境跑完一个episode性能的度量
                2. `abnormal`: 环境每个时间步都可包含该键值，该键值非必须，是可选键值，值为bool类型，表示环境运行该步是是否发生了错误，如果为真nervex的相关模块会进行相应处理（比如将相关数据移除）。


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

    1. 所有代码实现中命名建议一般情况使用单数，但如果使用复数可以使某局部代码块逻辑更清晰，该部分也可自由选择。
    2. 所有代码实现秉承 **自身对外界输入质疑，自身对外界输出负责** 的思想，对输入参数做必要的check，对输出（返回值）明确规定其格式
    3. 环境元素的键值如果为空时，一律使用 ``None``
