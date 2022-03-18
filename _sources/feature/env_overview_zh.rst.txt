Env Overview
===================


BaseEnv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(ding/envs/env/base_env.py)

概述：
    环境模块，是强化学习中重要的模块之一。在一些常见的强化学习任务中，例如 `atari <https://gym.openai.com/envs/#atari>`_ 相关任务，`mujoco <https://gym.openai.com/envs/#mujoco>`_ 相关任务，我们所训练的智能体就是在这样的环境中去进行探索和学习。通常，定义一个环境需要从环境的输入输出入手，并充分考虑其中可能的 ``observation space`` 与 ``action space``。OpenAI 所出品的 `gym` 模块已经帮我们定义了绝大多数常用的学术环境。`DI-engine` 也遵循 `gym.Env` 的定义，并在其基础上增加了一系列更为方便的功能，使得环境的调用更为简单易懂。

具体实现：
    我们可以很方便地查阅到，`gym.Env` 的 `定义 <https://github.com/openai/gym/blob/master/gym/core.py#L8>`_ 中，最为关键的部分在于 ``step`` 和 ``reset`` 方法。通过给定的 ``observation``, ``step()`` 方法依据输入的 ``action`` 并与环境产生交互，从而得到相应的 ``reward``。 ``reset()`` 方法则是对环境进行重置。关于环境中的 observation, action, reward，也给出了规范化的定义与限制，分别是 ``observation_space`` ``action_space`` 与 ``reward_range``。 `ding.envs.BaseEnv` 与 `gym.Env` 类似，除了上述接口与特性外，还定义并实现了：

    1. ``BaseEnvTimestep(namedtuple)``：定义了环境每运行一步返回的内容（即 ``step`` 的返回值），一般包括 ``obs`` ``act`` ``reward`` ``done`` ``info`` 五部分，子类可以自定义自己的该变量，但注意必须包含上述五个字段。

    2. space属性。保留了 ``observation_space`` 与 ``action_space`` 的定义，将 ``reward_range`` 扩展为 ``reward_space``，使其与前两个保持一致。如果是多智能体环境，还应当指明 ``num_agent``。

    .. note::

        ``subprocess_env_manager`` 中 ``shared_memory`` 强烈依赖 ``observation_space`` 的实现，如果要使用请务必保证 ``observation_space`` 的正确性。

    3. ``create_collector_env_cfg()``：为数据收集创建相应的环境配置文件，与 ``create_evaluator_env_cfg`` 互相独立，便于使用者对数据收集和性能评测设置不同的环境参数，根据传入的初始配置为每个具体的环境生成相应的配置文件，默认情况会获取配置文件中的环境个数，然后将默认环境配置复制相应份数返回

    4. ``create_evaluator_env_cfg()``：为性能评测创建相应的环境配置文件，功能同上说明

    5. ``enable_save_replay()``：使环境可以保存运行过程为视频文件，便于调试和可视化，一般在环境开始实际运行前调用。该方法一半只用于告知环境需要保存 replay 并指定路径。（该方法可选实现）

    此外，`ding.envs.BaseEnv` 还针对细节做了一些改动，例如

    1. ``seed()``: 对于环境内各种处理函数和 wrapper 的种子控制，外界设定的是种子的种子，当环境用于收集数据时，通常还会区分 **静态种子与动态种子**

    2. 默认都使用 **lazy init**，即 init 只设置参数，第一次 reset 时启动环境、设定种子

    3. episode 结束时，在 ``BaseEnvTimestep.info`` 这个 dict 中返回 ``final_eval_reward`` 键值对

    .. note::

        对于一个环境的具体创建（例如打开其他模拟器客户端），该行为不应该在 ``__init__`` 方法中实现，因为存在创建模型实例但不运行的使用场景（比如获取环境 observation 的维度等信息），推荐在 ``reset`` 方法中实现，即判断运行环境是否已创建，如果没有则进行创建再 reset，如果有则直接reset已有环境。如果使用者依然想要在 ``__init__`` 方法中完成该功能，请自行确认不会有资源浪费或冲突的情况发生。

    .. note::

        关于 ``BaseEnvTimestep``，如无特殊需求可以直接调用 `DI-engine` 提供的默认定义，即：

        .. code:: python

            from ding.envs import BaseEnvInfo

        如果需要自定义，按照上文的要求使用 ``namedtuple(BaseEnvTimestep)`` 实现即可。

    .. tip::

        ``seed`` 方法的调用一般在 ``__init__`` 方法之后，``reset`` 方法之前。如果将模型的创建放在 ``reset`` 方法中，则 ``seed`` 方法只需要记录下这个值，在 ``reset`` 方法执行时设置随机种子即可。

    .. warning::

        `DI-engine` 对于环境返回的 ``BaseEnvTimestep.info`` 字段有一些依赖关系, ``BaseEnvTimestep.info`` 是一个 dict，其中某些键值对会有相关依赖要求：
        
        1. ``final_eval_reward``: 环境一个 episode 结束时（ ``done==True``时 ）必须包含该键值，值为 float 类型，表示环境跑完一个 episode 性能的度量
        
        2. ``abnormal``: 环境每个时间步都可包含该键值，该键值非必须，是可选键值，值为 bool 类型，表示环境运行该步是是否发生了错误，如果为真 `DI-engine` 的相关模块会进行相应处理（比如将相关数据移除）。


    类继承关系如下图所示：
    
        .. image:: images/baseenv_class_uml.png
            :align: center
            :scale: 60%
