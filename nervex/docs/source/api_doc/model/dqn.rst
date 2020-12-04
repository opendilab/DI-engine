model.dqn
===================

dqn network
-----------------

DQNBase
~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.model.dqn.dqn_network.DQNBase
    :members: __init__, forward, fast_timestep_forward

Encoder
~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.model.dqn.dqn_network.Encoder
    :members: __init__, forward

Head
~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.model.dqn.dqn_network.Head
    :members: __init__, forward

parallel_wrapper
~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.model.dqn.dqn_network.parallel_wrapper

get_kwargs
~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.model.dqn.dqn_network.get_kwargs