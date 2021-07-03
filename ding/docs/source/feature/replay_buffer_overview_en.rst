Replay Buffer Overview
========================

IBuffer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(ding/worker/replay_buffer/base_buffer.py)

Overview:
    ``IBuffer`` is an abstract interface. All types of replay buffers should implement all abstract methods of ``IBuffer``. 

Variables:
    None

Abstract class interface method: (All but ``default_config`` should be overroidden by subclasses)
    1. ``default_config`` (classmethod): Default config of this buffer.
    2. ``push``: Push data into this buffer.
    3. ``update``: Update some info of this buffer, e.g. some data's priority.
    4. ``sample``: Sample given size datas.
    5. ``clear``: Clear all data.
    6. ``count``: Count valid data number.
    7. ``state_dict``: Save state dict of this buffer.
    8. ``load_state_dict``: Load state dict to this buffer.

NaiveReplayBuffer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(ding/worker/replay_buffer/naive_buffer.py)

Overview:
    ``NaiveReplayBuffer`` is a naive implementation. It is a First-In-First-Out cicular queue with random sampling. And it does not have any monitor or logger either.

Variables:
    replay_buffer_size, push_count

Class interface method: (All should be overroidden by subclasses)
    1. ``__init__``: Initialize with config.
    2. ``push``: Push data at the tail of the circular queue
    3. ``update``: No info to update, but this method is preserved for compatibility.
    4. ``sample``: **Randomly** sample some datas.
    5. ``clear``: Clear all data.
    6. ``count``: Count valid data number.
    7. ``state_dict``: Save state dict of this buffer.
    8. ``load_state_dict``: Load state dict to this buffer.
    9. ``start``: Start `UsedDataRemover`. Details can be found in **Full data & Meta data**
    10. ``close``: Close `UsedDataRemover`. Details can be found in **Full data & Meta data**

Full data & Meta data
------------------------
In DI-engine, we define **full data** and **meta data**.

**Full data** is often a dict, with keys ``['obs', 'action', 'next_obs', 'reward', 'info']`` and some optional keys like ``['priority', 'use_count', 'collect_iter', ...]``. However, in some complex environments(Usually we run them in parallel mode), full data can be too big to store in memory. Therefore, we divide full data into **file data** and **meta data**.

**File data** is usually too big to store in memory, therefore is stored in file system. **meta data** includes ``'file_path'`` and some keys which can be used in sampling. **meta data** is usually small, so it can easily be stored in replay buffer(in memory). 

Therefore, in parallel mode, when removing the data out of buffer, we must not only remove meta data in memory, but also remove corresponding file data in the file system as well. DI-engine uses ``UsedDataRemover`` (ding/worker/replay_buffer/utils.py) to track and remove used file data.

This mechanism is adopted in **all buffers**.


AdvancedReplayBuffer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(ding/worker/replay_buffer/advanced_buffer.py)

Overview:
    ``AdvancedReplayBuffer`` is a buffer with more advanced features, e.g. Prioritized Sampling, Data Quality Monitor, Thruput Control, Logger. It is generally a First-In-First-Out cicular queue, but due to advanced features, its remove, sample, update are more complicated.

Variables:
    beta, replay_buffer_size, push_count

Class interface method: (All should be overroidden by subclasses)
    1. ``__init__``: Initialize with config.
    2. ``push``: Push data at the tail of the circular queue
    3. ``update``: Update datas' **priorities**.
    4. ``sample``: Sample some datas according to **priority**.
    5. ``clear``: Clear all data.
    6. ``count``: Count valid data number.
    7. ``state_dict``: Save state dict of this buffer.
    8. ``load_state_dict``: Load state dict to this buffer.
    9. ``start``: Start `UsedDataRemover`.
    10. ``close``: Close `UsedDataRemover`, **monitor** and **logger**.

Here is an image demonstrating "push", "remove", "sample" procedures of ``AdvancedReplayBuffer``, including all advance features. Then we will introduce these features one by one.

.. image::
   ../key_concept/images/advanced_buffer.png
   :align: center
   :scale: 65%

Feature1: Prioritized Sampling
--------------------------------
Implement `Prioritized Experience Replay <https://arxiv.org/abs/1511.05952>`_.
Use segment tree to

    - Store each data's priority (sample propability)
    - Sample according to priority
    - Support priority update 

DI-engine also uses **numba** to optimize profile.

Feature2: Data Quality Monitor
--------------------------------
Monitor two data quality attributes: staleness and use_count. They both function during sampling to control sampled data quality.

**staleness** measures model iteration gap between the time when it is collected and the time when it is sampled. If the gap is too big, it means the data is too stale to use, so the data will be removed in ``sample_check``.

**use_count** count how many times a piece of data is sampled. If when a data is sampled and this time it reaches the max limit, the data will be removed so it will not be sampled again, but this time it can be used. This mechanism affects "UsedDataRemover": This time it can be sampled and used, but after that it should be removed out of buffer. However, if we remove it at once, learner might not be able to find the file data when training. As a result, we set its priority to 0 instead, and it will be removed when the circular queue's head points to it.


Feature3: Throughput Control
--------------------------------
In serial mode, we can modify ``n_sample`` or ``n_episode`` to control how many samples or episodes to collect in collector's turn; And modify ``batch_size`` and ``update_per_collect`` to control how many samples are used in learner's turn.

However, in parallel mode, it is more complicated to balance the data speed in collector end and learner end. Therefore, we use ``ThruputController`` (ding/worker/replay_buffer/utils.py) to limit the "push" / "sample" rate in a [min, max] range.

Also, user can set parameter ``sample_min_limit_ratio`` to control the min ratio of "valid count" / "batch_size". If there are not enough valid datas, buffer can refuse to sample. 

Feature4: Logger
--------------------------------
Create tensorboard logger and text logger, to record data quality attributes(Feature2) and throughput statistics(Feature3). This feature is very useful in debugging and tuning.



EpisodeReplayBuffer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(ding/worker/replay_buffer/episode_buffer.py)

In some scenarios, a whole episode is of bigger use than separated samples, such as chess, card games or some specific algorithms like `Hindsight Experience Replay <https://arxiv.org/abs/1707.01495>`_. Therefore, we need a buffer, where each element is no longer a training sample, but an episode. Currently, DI-engine ``EpisodeReplayBuffer`` is derived from ``NaiveReplayBuffer``, because they two share so many common features. 
However, they two have two main differences.

The **first** one is: Each element is a whole episode, rather than a sample.

In addition, when sampling out episodes, sometimes an algorithm does not require a whole episode, but a fixed length within several episodes. As a result:

The **second** one is: (Maybe) Do not sample `batch_size` elements(episodes). Instead, first do some operations, then return some chruncated train samples.



          