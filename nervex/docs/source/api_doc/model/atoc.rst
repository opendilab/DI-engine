model.atoc
===================

atoc_network
-----------------

ATOCAttentionUnit
~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.model.atoc.atoc_network.ATOCAttentionUnit
    :members: __init__, forward

ATOCCommunicationNet
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.model.atoc.atoc_network.ATOCCommunicationNet
    :members: __init__, forward


ATOCActorNet
~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.model.atoc.atoc_network.ATOCActorNet
    :members: __init__, forward


ATOCCriticNet
~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.model.atoc.atoc_network.ATOCCriticNet
    :members: __init__, forward


ATOCQAC
~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.model.atoc.atoc_network.ATOCQAC
    :members: __init__, forward, compute_q, compute_action, optimize_actor, optimize_actor_attention

