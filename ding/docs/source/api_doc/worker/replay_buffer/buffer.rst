worker.replay_buffer
======================

replay buffer
-------------------

IBuffer
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.replay_buffer.base_buffer.IBuffer
    :members: push, update, sample, clear, count, state_dict, load_state_dict, default_config

NaiveReplayBuffer
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.replay_buffer.naive_buffer.NaiveReplayBuffer
    :members: start, close, push, update, sample, clear, count, state_dict, load_state_dict, default_config

AdvancedReplayBuffer
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.replay_buffer.advanced_buffer.AdvancedReplayBuffer
    :members: start, close, push, update, sample, clear, count, state_dict, load_state_dict, default_config

EpisodeReplayBuffer
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.replay_buffer.episode_buffer.EpisodeReplayBuffer
    :members: __init__, start, close, push, update, sample, clear, count, state_dict, load_state_dict, default_config

create_buffer
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: ding.worker.replay_buffer.base_buffer.create_buffer

get_buffer_cls
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: ding.worker.replay_buffer.base_buffer.get_buffer_cls

utils
-------------------

UsedDataRemover
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.replay_buffer.utils.UsedDataRemover
    :members: start, close, add_used_data

SampledDataAttrMonitor
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.replay_buffer.utils.SampledDataAttrMonitor

PeriodicThruputMonitor
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.replay_buffer.utils.PeriodicThruputMonitor
