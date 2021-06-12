worker.replay_buffer
======================

replay buffer
-------------------

IBuffer
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.replay_buffer.base_buffer.IBuffer
    :members: push, update, sample, clear, count, state_dict, load_state_dict

NaiveReplayBuffer
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.replay_buffer.naive_buffer.NaiveReplayBuffer
    :members: __init__, start, close, push, update, sample, clear, count, state_dict, load_state_dict

AdvancedReplayBuffer
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.replay_buffer.advanced_buffer.AdvancedReplayBuffer
    :members: __init__, start, close, push, update, sample, clear, count, state_dict, load_state_dict

EpisodeReplayBuffer
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.replay_buffer.episode_buffer.EpisodeReplayBuffer
    :members: __init__, start, close, push, update, sample, clear, count, state_dict, load_state_dict