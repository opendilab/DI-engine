Buffer Manager Overview
========================

概述：
    Prioritized Replay Buffer（优先回放经验池）是强化学习中的一个核心概念，用于将collector收集到的数据存储在其中，并在learner需要训练数据时
    按照一定的优先级进行采样，可以有效解决样本存在相关性而不满足独立同分布假设、样本使用效率过低等问题，保证训练效果更加稳定。

    我们的Buffer Manager可以包括多个Replay Buffer，manager可以指定将数据存在哪个buffer，也可以按照一定的策略从不同的buffer中采样训练样本，
    即manager承担了管理多个buffer、统筹调度的工作。诚然，大部分算法使用一个replay buffer就可以实现，也无需manager进行管理，
    但近些年随着越来越多的强化学习论文与研究着眼于数据流，相信我们的拥有更强可扩展性的设计可以极大地方便研究者们。

Episode buffer
Also, when sampling out episodes, sometimes an algorithm does not require a whole episode, but a fixed length within several episodes. So in ``EpisodeReplayBuffer``, the main differences from normal sample buffer(like ``NaiveReplayBuffer`` or ``AdvancedReplayBuffer``) are as follows:

   1. Each element is a whole episode, rather than a sample.
   2. (Maybe) Do not sample `batch_size` elements(episodes), however, first do some operations, then return 


Buffer Manager
--------------------

概述：
    Buffer Manager可以管理一个或多个Replay Buffer。manager的线程安全由buffer保证。

    我们建议外界通过manager的接口来和buffer进行交互，而不直接与buffer进行交互。 ``push_data`` 接收传入的数据与指定的buffer名，向指定buffer插入数据；
    ``sample`` 按照一定的策略从所有buffer中采样数据； ``update`` 更新对应buffer的信息； ``close`` 关闭全部buffer。
    
代码结构：
    相关的类有两个：

        1. BufferManager(nervex/data/buffer_manager.py): replay buffer主体。
        2. SumSegmentTree(nervex/data/structure/segment_tree.py): 用于各个buffer的采样个数确认。

接口说明：
    Buffer Manager暴露给外界的主要接口分为：存储数据、采样数据、数据元信息更新。

    1. push_data

        - 概述：
            
            接收的参数为插入数据和希望插入的buffer，然后manager就可以调用对应buffer的 ``extend`` 方法将数据插入。

    2. sample

        - 概述：

            若接收到了 ``sample_ratio`` 参数，则要根据其相应调整采样策略，否则将按照config中的进行采样。

            各个buffer的sample ratio可以类比每个buffer中各个数据的priority，可理解为选中的概率，故也采用线段树进行选择。
            首先决定本次sample出的list的每个元素需要从哪个buffer得到，然后去对应的buffer调用 ``sample`` 方法进行采样，最后再将数据塞到各个位置并返回。

    3. update

        - 概述：

            manager接收到的信息是所有buffer混在一起的， 利用形式为"BufferName_DataId"的“replay_unique_id”分离得到buffer名，
            区分出buffer各自的更新信息，然后调用每个buffer的 ``update`` 方法进行更新。


Replay Buffer
--------------------

概述：
    Replay Buffer是Replay Buffer进行数据存储、采样的基本单位。buffer是线程安全的实现，在所有对buffer内数据有读写操作时都有线程锁以保护线程安全。

    我们建议使用buffer时，应由manager调用接口并由manager统一管理。 ``append`` 和 ``extend`` 是将数据送入buffer的接口，区别是 ``append`` 传入单个数据，
    而 ``extend`` 传入 ``list`` 存放的多个数据。 ``sample`` 是从buffer采样数据的接口，但在 ``sample`` 前必须调用 ``sample_check``
    进行采样检测。 ``clear`` 可以将buffer清空并重置为初始状态。 ``close`` 可以正确关闭buffer。


代码结构：
    相关的类有以下几个：

        1. AdvancedReplayBuffer(nervex/data/structure/buffer.py): buffer主体。
        2. RecordList(nervex/data/structure/buffer.py): 继承自list，用于记录buffer中由于更新而被移除的旧数据。
        3. NaturalMonitor, InTickMonitor, OutTickMonitor(nervex/data/structure/buffer.py): 来自autolog，用于记录buffer统计数据。
        4. SegmentTree, SumSegmentTree, MinSegmentTree(nervex/data/structure/segment_tree.py): 用于buffer中的prioritized sample。

接口说明：
    Replay Buffer暴露给Buffer Manager的主要接口分为：存储数据、采样数据、数据元信息更新。

    1. append & extend

        - 概述：

            由manager的 ``push_data`` 调用。 
            
            将collector产生的数据存入buffer，二者的区别在于数据是单个还是一个list包含多个。首先要调用 ``_data_check`` 对数据进行检查，
            通过检查的数据会被增添 ``['replay_unique_id', 'replay_buffer_idx']`` 两个域，分别代表该数据的唯一身份标识
            和其在buffer中的索引。最后，更新monitor与logger。

    2. sample_check & sample

        - 概述：

            由manager的 ``sample`` 调用，在调用时，必须先调用 ``sample_check`` 进行检查和预处理，然后才能调用 ``sample`` 进行实际的采样。

            ``sample_check`` 会先计算buffer中数据的staleness，其等于数据被collector采集到的model iteration和现在learner需要采样数据时的model iteration之间的差值。
            如果某条数据的staleness超过一定限度，会将该条数据从buffer中移除。
            
            若 ``sample_check`` 结果为True，即通过检查，才会触发后续被调用``sample`` 。
            buffer利用线段树实现了prioritized sample，将priority存储在线段树中以进行采样的原理此处不做详细说明，总之，buffer可以将priority作为概率进行采样。
            首先调用 ``_get_indices`` 得到需要被采样的所有索引值（这一步便用到了上一句中的线段树），然后调用 ``_sample_with_indices`` 实际来取出数据。
            然后，计算每一条数据的staleness, use, importance sampling weight并附加到其域中。
            如果发现sample到了相同的数据，则需要deepcopy以保证数据安全性。
            如果发现该数据的use超过一定限度，会将该条数据从buffer中移除。

            最后，更新monitor与logger。

    3. update

        - 概述：

            由manager的 ``update`` 调用。

            传入的更新信息为dict，包括 ``['replay_unique_id', 'replay_buffer_idx', 'priority']`` 这几个域。
            利用该信息即可在线段树中更新一条数据的priority。
