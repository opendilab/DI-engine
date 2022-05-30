template.qmix
-------------------------------------------------------

Please Reference ding/model/template/qmix.py for usage

Mixer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.template.qmix.Mixer
    :members: __init__, forward

QMix
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.template.qmix.QMix
    :members: __init__, forward, _setup_global_encoder

CollaQMultiHeadAttention
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.template.qmix.CollaQMultiHeadAttention
    :members: __init__, forward

CollaQSMACAttentionModule
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.template.qmix.CollaQSMACAttentionModule
    :members: __init__, _cut_obs, forward


CollaQ
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.template.qmix.CollaQ
    :members: __init__, forward, _setup_global_encoder
