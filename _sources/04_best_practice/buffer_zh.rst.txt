Buffer 使用指南
===============================

Buffer 入门
-------------------------------

在 Off-policy RL 算法中，我们通常会使用经验回放（Experience Replay）机制来提高样本利用效率，并降低样本之间的相关性。
DI-engine 提供了 \ **DequeBuffer** \ 来实现经验回放池的常见功能，例如数据的存入、采样等。用户可以通过以下命令创建 DequeBuffer 对象:

**Buffer 的基本概念**

.. code-block:: python

    from ding.data import DequeBuffer

    buffer = DequeBuffer(size=10)


在 DI-engine 的 buffer 以及一些其它组件中，我们使用数据类（dataclass）作为数据的结构载体。
Dataclass 是 python3 的一种特性，相比于字典（Dict），它能够明确规定类中某个字段（类的属性）的数据类型，使得数据整齐一致；
相比于命名元组（Namedtuple），它可以通过设置默认值，在初始化阶段实现参数的缺省，也可以在使用过程中完成灵活的赋值操作。
以下，我们将为用户介绍 buffer 的具体操作方式。


.. code-block:: python

    # 数据存入，每次处理一条样本。
    # 在 DI-engine 的中间件中，缓存数据类型通常为字典，记录样本的 obs，next_obs，action，reward 等信息。
    for _ in range(10):
        # BufferedData 对象包含 data，index 和 meta 三个字段。
        # 其中 data 为待缓存数据本身，meta 为其元信息（可选， 默认为 None），这两项通过 push 方法传入 buffer。
        # index 表示该数据在 buffer 中实际存储地址的索引，由 buffer 自动生成，不需要用户手动设置。
        buffer.push('a', meta={})

    # 数据采样每次处理多条样本，用户需要明确指定采样的数量，参数 replace 表示采样时是否放回，默认值为 False。
    # 采样操作返回一个名叫 BufferedData 的数据类对象，例如：BufferedData(data='a', index='67bdfadcd', meta={})
    buffered_data = buffer.sample(3, replace=False)
    data = [d.data for d in buffered_data]


**使用 Buffer 完成在线训练**

在上一小节中，我们介绍了 buffer 中数据的实际存储结构，以及最基本的存入与采样操作。
事实上，在大部分任务中，用户并不需要使用这些底层的原子操作。我们\ **更加推荐**\用户通过 DI-engine 封装好的中间件来调用该 buffer 对象来完成训练。

.. code-block:: python
    
    from ding.framework import task
    from ding.framework.middleware import data_pusher, OffPolicyLearner

    task.use(data_pusher(cfg, buffer))
    task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer))


**使用 Buffer 加载专家数据**

在 SQIL、DQFD 这类模仿学习任务中，我们需要使用专家数据参与训练。
这时，用户可以额外使用一个 buffer 来加载专家数据，以 SQIL 为例（完整版代码可见 \ `./ding/example/sqil.py <https://github.com/opendilab/DI-engine/blob/main/ding/example/sqil.py>`_）：

.. code-block:: python
    
    from ding.framework.middleware import sqil_data_pusher

    buffer = DequeBuffer(size=10)
    expert_buffer = DequeBuffer(size=10)

    task.use(sqil_data_pusher(cfg, buffer_=buffer, expert=False))
    task.use(sqil_data_pusher(cfg, buffer_=expert_buffer, expert=True))
    task.use(OffPolicyLearner(cfg, policy.learn_mode, [(buffer, 0.5), (expert_buffer, 0.5)]))


Buffer 进阶
-------------------------------

在上一节中，我们提供了 buffer 基本的应用场景。接下来，我们将为用户深入展示 buffer 更全面的功能。


**优先级采样**

在一些算法中，需要用到优先级采样。在 DI-engine 中，使用 \ **PriorityExperienceReplay 中间件**\，即可赋予 buffer 优先级采样功能。
如果用户使用了该功能，在存入样本时，用户还必须在 meta 中补充样本的优先级信息，如下所示。\ **优先级采样会增加采样耗时**\。

.. code-block:: python
    
    from ding.data.buffer.middleware import PriorityExperienceReplay

    buffer = DequeBuffer(size=10)
    buffer.use(PriorityExperienceReplay(buffer, IS_weight=True))
    for _ in range(10):
        # meta 的本质为一个字典，用于补充对样本的描述，默认为空。
        buffer.push('a', meta={"priority": 2.0})
    buffered_data = buffer.sample(3)


**样本克隆**

在默认情况下，对于 buffer 中存储的可变对象（如 list、np.array、torch.tensor 等），采样操作事实上是返回了对该对象的引用。
如果用户后续对该引用的内容进行了修改，可能会导致样本池中的对应内容也发生变化。
在某些应用场景上，用户可能期望样本池中的数据保持不变，这时就可以通过使用 \ **clone_object 中间件**\，在采样时返回 buffer 中对象的拷贝。
这样一来，对拷贝内容的修改就不会影响到 buffer 中的原数据。\ **样本克隆会显著增加采样耗时**\。

.. code-block:: python
    
    from ding.data.buffer.middleware import clone_object

    buffer = DequeBuffer(size=10)
    buffer.use(clone_object())


**分组采样**

在某些特殊环境或算法中，用户可能希望以整个剧集 (episode) 为单位收集、存储和处理样本。
例如：在国际象棋、围棋或纸牌游戏中，玩家只有在游戏结束时才能获得奖励，解决这类任务时算法往往希望对整局游戏进行处理，此外像 Hindsight Experience Replay (HER) 等一些算法需要采样完整的 episode，并以 episode 为单位进行相关处理。
这时，用户可以使用分组采样的方式实现这一目标。

- **通过原子操作自定义实现**

  存储样本时，用户可以在 meta 补充 "episode" 信息，以明确样本所属的 episode。采样时，通过设定 groupby="episode" 即可来实现以 episode 为关键字的分组采样。\ **分组采样会严重增加采样耗时**\。

  .. code-block:: python

      buffer = DequeBuffer(size=10)

      # 存入数据时，用户需要在 meta 中补充分组信息，如：以 "episode" 为分组关键字，相应的值则为具体的组别
      buffer.push("a", {"episode": 1})
      buffer.push("b", {"episode": 2})
      buffer.push("c", {"episode": 2})

      # 根据关键字 "episode" 来分组，需要保证 buffer 中不同的组的数量不小于采样数量。
      grouped_data = buffer.sample(2, groupby="episode")

- **通过中间件实现**

  以 R2D2 算法为例，训练样本以 episode 为单位经过 LSTM 网络，这里就需要用到分组采样。
  在 DI-engine 中，每个 env 对应一条独立的决策轨迹，因此使用 env 作为分组的关键字就能够起到区分 episode 的效果。
  相关代码示例如下，完整版代码可见 \ `./ding/example/r2d2.py <https://github.com/opendilab/DI-engine/blob/main/ding/example/r2d2.py>`_：

  .. code-block:: python

      buffer = DequeBuffer(size=10)

      # 这里将 'env' 作为分组的关键字，在采样时，同一个 env_id 的样本将会被划分到同一个 group 中。
      task.use(data_pusher(cfg, buffer, group_by_env=True))


**(可选项)**
在分组采样的基础上，还可以再使用 \ **group_sample中间件**\ 实现样本的后处理工作，如：选择是否打乱同组内数据，以及设定每组数据的最大长度。

.. code-block:: python
    
    from ding.data.buffer.middleware import group_sample

    buffer = DequeBuffer(size=10)
    # 每组数据的最大长度设为3，保持组内原顺序
    buffer.use(group_sample(size_in_group=3, ordered_in_group=True))
    

**删除多次使用样本**

在默认条件下，buffer 中的样本有可能会被重复多次采集。在训练过程中，如果不加控制地反复使用这些重复样本，可能会导致模型的效果不佳。
为了避免这个问题，我们可以使用 \ **use_time_check 中间件**\ 来设置样本的最大使用次数。

.. code-block:: python
    
    from ding.data.buffer.middleware import use_time_check

    buffer = DequeBuffer(size=10)
    # 设置单条样本的最大使用次数为2
    buffer.use(use_time_check(buffer, max_use=2))

实际采样时，该中间件会维持一个样本使用次数的计数，并将其写入 meta 中的 use_count 字段。
当使用计数大于等于 buffer 设定的最大次数时，该样本将会被永久删除。
