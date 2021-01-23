model.common
===================

actor_critic
-----------------

ValueActorCriticBase
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.model.common.actor_critic.ValueActorCriticBase
    :members: forward, seed, compute_action, compute_action_value, mimic


QActorCriticBase
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.model.common.actor_critic.QActorCriticBase
    :members: forward, seed, optimize_actor, compute_action, compute_q, mimic

SoftActorCriticBase
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.model.common.actor_critic.SoftActorCriticBase
    :members: forward, seed, evaluate, compute_action, compute_q, compute_value, mimic



dueling
--------------------

DuelingHead
~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.model.common.dueling.DuelingHead
    :members: __init__, forward


encoder
--------------------

ConvEncoder
~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.model.common.encoder.ConvEncoder
    :members: __init__, forward

FCEncoder
~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.model.common.encoder.FCEncoder
    :members: __init__, forward
