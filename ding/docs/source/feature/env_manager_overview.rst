Env Manager Overview
========================


Env Manager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

概述：
    env manager 是一个环境管理器，可以管理多个相同类型不同配置的环境。env manager 可以实现多个 env 同时运行，同时获取环境中的信息，并且提供与 env 相似的接口，可以大大简化 code，加速运行。
    目前支持的类型有单进程串行和多进程并行两种模式。BaseEnvManager 通过循环串行（伪并行）来维护多个环境实例，Async(Sync)SubprocessEnvManager 通过子进程向量化的方式，即调用
    multiprocessing，通过在子进程中运行 env，以进程间通信的方式对环境进行管理和运行。DI-engine 的 env manager 需使用 DI-engine 格式的 env 定义（或者由 EnvWrapper装饰过的 Gym env)，
    其初始化时需提供每个 env 的实例化接口，通过 config 设定具体的运行细节。

    一般来说，:class:`BaseEnvManager <ding.envs.BaseEnvManager>` 用于一些简单环境的运行或 debug，复杂环境或大数量环境的运行推荐采用 
    :class:`SyncSubProcessEnvManager <ding.envs.SyncSubProcessEnvManager>` 和 :class:`AsyncSubProcessEnvManager <ding.envs.AsyncSubProcessEnvManager>` 进行加速。

    如果对 env 模块还不够了解，建议先查阅 DI-engine 的 `Env Overview <./env_overview.html>`_

用法：
    - init
        env manager 的初始化需要传入每个 env 的实例化调用接口和 config 字典，可通过 lambda 函数或者偏函数 ``functools.partial`` 来对 env 的实例化函数进行包装，指定其运行参数。


        .. code:: python

            config = dict(
                env=dict(
                    manager=dict(...),
                    ...
                ),
                ...
            )

            # lambda function way
            env_fn = lambda : DI-engineEnv(*args, **kwargs)
            env_manager = BaseEnvManager(env_fn=[env_fn for _ in range(4)], cfg=config.env.manager)

            # partial function way
            from functools import partial
            
            def env_fn(*args, **kwargs):
                return DI-engineEnv(*args, **kwargs)
            env_manager = BaseEnvManager(env_fn=[partial(env_fn, *args, **kwargs) for _ in range(4)], cfg=config.env.manager)

    - launch/reset
        env manager 初始化后并不会立即实例化每个环境，此时 env manager 会被标记为 `closed` 状态。首次初始化环境需调用 ``launch`` 方法，该方法会按照传入的 env 实例化调用接口
        构造每个 env 实例（对 SubprocessEnvManager 来说则是运行每个环境的子进程，建立通信管道），构造一些环境运行时的状态变量等，同时调用各子环境的 ``reset`` 方法，将环境运行起来。

        .. warning::

            调用在 `closed` 状态的 env_manager 的 ``step`` 和 ``reset`` 方法会引发异常。

        在调用过 ``launch`` 方法之后便可通过调用 env manager 的 ``reset`` 方法来手动 reset 子环境。当不传入任何参数时，默认会 reset 所有子环境。当传入 ``reset_param`` 
        参数时，会 reset ``reset_param`` 中的键对应的子环境，并将其键值作为子环境 ``reset`` 方法的参数。由于不确定每个子环境 reset 需要的时间，env manager 不会返回子环境的 step
        运行结束后对应的 observation，而是会在 reset 结束时将返回值保存起来，通过调用 ``ready_obs`` 属性获得当前运行完成 step 或 reset 方法的子环境的
        observation，此举可以加快 SubprocessEnvManager 的运行效率。
        
        .. note::

            当 SubprocessEnvManager 需要 reset 正在进行 reset 的子环境时，该方法会等待这些子环境的上一次 reset 运行完毕再运行此次 reset。

    - step
        step 方法会串行地（BaseEnvManager）或并行地（SubprocessEnvManager）调用 env manager 中子环境的 step 方法，并返回 step 的结果，将 observation 存入 ``ready_obs``
        属性中。该方法传入的参数是一个 ``actions`` 字典，其键指定了需要运行 ``step`` 的 env_id，键值为该子环境的 ``step`` 运行的 action。依据不同的 env manager 类型和 config 设置，
        当有一定数量的子环境返回 step 结果后，该方法会检查运行结果，根据这些结果修改子环境的运行状态，并返回结果或抛出异常。

        .. warning::

            ``actions`` 包含正在运行其他命令或已经完成 episode 的子环境 id 时会引发异常。
    
    - ready_obs
        ``ready_obs`` 属性返回一个字典，内容为环境的 env_id 和最新返回的 observation 的键值对。对 SubprocessEnvManager 来说 ``ready_obs`` 属性返回的环境 id 一定是完成了 reset
        或 step 方法，正在等待新命令的子环境，因此可以安全地继续调用这些子环境的 ``reset`` 和 ``step`` 方法。当所有仍在运行（未运行至 done）的子环境都没有完成 ``reset`` 和 ``step`` 
        方法的运行时，调用 ``ready_obs`` 属性会等待至少一个子环境完成运行，并返回其 observation。

        在使用 SubprocessEnvManager 时，只要给 step 和 reset 方法传入参数的 env_id 均是来自 ready_obs 属性返回的 env_id ，就不会出现为子环境重复发送命令的情况。
    
    - done
        该属性会判断所有子环境的完成情况（是否运行至 done），若是，返回 ``True``，否则返回 ``False``
    
    - close
        同 Gym env 的 ``close`` 方法一样，该方法会安全地关闭所有的子环境，销毁子环境开辟的进程，释放全部资源。调用该方法后，env manager 会被标记为 ``closed``，除非重新 ``launch``
        才能继续使用。

样例：
    以下为一个 env manager 运行多个环境的实例

    .. code:: python

        my_env_manager.launch()

        while not finished:
            obs = my_env_manager.ready_obs
            actions = ... # get actions from policy or else.
            timesteps = my_env_manager.step()
            for env_id, timestep in timesteps.item():
                if timestep.done:
                    # without auto_reset
                    my_env_manager.reset(reset_param={env_id: ...})
                    ...

        my_env_manager.close()

高级特性：
    - auto_reset
        
        DI-engine 的 env manager 默认会进行自动 reset，即当某个环境运行至 done 之后会自动 reset 以继续运行，reset 时的参数为上次手动 reset 时为该子环境设置的参数，
        除非累计运行的 episode 数量达到 config 中指定的 episode_num。若要关闭该特性,可在 config 中指定 ``auto_reset=False``

    - env state

        为方便管理各子环境的状态并便于 debug，DI-engine 的 env manager 提供了环境状态的枚举类型来实时掌握所有子环境的运行状态，其具体含义如下：

        - VOID: 初始化了 env manager，尚未实例化子环境
        - INIT: 实例化了子环境，尚未进行 launch 或 reset
        - RUN: 完成了 reset 或 step ，正在运行中的子环境
        - RESET: 正在进行 reset 的子环境
        - DONE: 运行至 done 的子环境
        - ERROR: 发生异常的子环境
        
        各状态间的转换关系如图示：

            .. image:: images/env_state.png

    - max_retry 和 timeout
  
        为防止有些子环境因连接问题短暂地报错，或子进程卡死时程序不会正常退出，DI-engine 的 env manager 添加了 retry 保护和 timeout 检测机制。用户可在 config 中指定最大 retry 次数，
        和 reset、step、 子进程间通信的最大等待时间，当超过等待时间时会抛出异常，以便提前终止运行。config 中这些参数的设置和默认值如下：

        .. code-block:: python

            manager_config = dict(
                max_retry=1, # step 和 reset 的最大重试次数，默认为 1
                reset_timeout=60, # reset 方法的等待时间，默认为 60s
                retry_waiting_time=0.1, # reset 方法 retry 的间隔时间，默认为 0.1s
                step_timeout=60, # step 方法的等待时间，默认为 60s
                step_wait_timeout=0.01, # step 方法 retry 的间隔时间，默认为 0.01s
                connect_timeout=60, # 子进程之间通信的等待时间，默认为 60s
            )

    - Sync 和 Async SubprocessEnvManager 的区别
  
    - shared_memory
        shared_memory 可以加速传递环境返回的大向量数据，当环境返回的obs等变量大小超过100kB时，推荐设置为True。使用shared_memory时，需要在环境info函数中，用BaseEnvInfo和EnvElementInfo template来指定对应obs、act和rew的shape和value的dtype。
  
    - get_attribute


BaseEnvManager (ding/envs/env_manager/base_env_manager.py)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

概述：
    使用循环串行的方式运行多个环境的管理器。

类接口方法：
    1. __init__: 初始化
    2. launch: 初始化所有子环境，初始化子环境状态管理所需的资源
    3. reset: 不传入参数时默认 reset 所有环境，传入 dict 结构的 env_id 和 reset_param 时，对 env_id 所指定的子环境按照 reset_param 进行 reset，并在运行结束时返回
    4. step: 环境执行输入的动作，完成一个时间步，同 reset 一样，可以传入 dict 结构的 env_id 和 action 对某几个环境进行操作，返回全部运行结果
    5. seed: 设置环境随机种子，可以传入 list 结构的 env_id 对 manager 持有的某几个环境设置特定的 seed
    6. close: 关闭环境，释放资源，close 所有环境

类属性方法：
    1. env_num: manager 中子环境的数量
    2. active_env: 所有未运行结束的环境 list
    3. ready_obs: 返回所有未运行结束的环境 env_id 和最新返回的 observation
    4. done: 是否所有持有的环境已经运行结束

SubprocessEnvManager (ding/envs/env_manager/subprocess_env_manager.py)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

概述：
    继承了BaseEnvManager，通过 multiprocessing 模块为每个环境创建单独的进程，使用并行的方式运行多个环境的管理器。

类接口方法：
    以下只列出与 BaseEnvManager 不同或新增的方法

    1. launch: 初始化运行每个子环境的进程，初始化子环境状态管理所需的资源
    2. reset: 不传入参数时默认 reset 所有环境，传入 dict 结构的 env_id 和 reset_param 时，对 env_id 所指定的子环境进程按照 reset_param 发送 reset 命令
    3. step: 为环境进程发送动作命令，同 reset 一样，可以传入 dict 结构的 env_id 和 action 对某几个环境进行操作，待全部或部分环境运行结束时返回结果
    4. close: close 所有环境，销毁环境子进程，释放资源

类属性方法：
    以下只列出与 BaseEnvManager 不同或新增的属性

    1. ready_obs: 返回完成了上一个 step 或 reset 命令的子环境 env_id 和返回的 observation，若所有环境均在运行上一个命令，等待直到至少一个环境返回了运行结果
    2. active_env: 所有在运行状态的环境 list
