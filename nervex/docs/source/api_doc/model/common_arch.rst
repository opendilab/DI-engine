model.common_arch
===================

actor_critic
-----------------

ValueActorCriticBase
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.model.common_arch.actor_critic.ValueActorCriticBase
    :members: forward, seed, compute_action, compute_action_value, mimic


QActorCriticBase
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.model.common_arch.actor_critic.QActorCriticBase
    :members: forward, seed, optimize_actor, compute_action, compute_q, mimic



dueling
--------------------

DuelingHead
~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.model.common_arch.dueling.DuelingHead
    :members: __init__, forward


encoder
--------------------

ConvEncoder
~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.model.common_arch.encoder.ConvEncoder
    :members: __init__, forward
