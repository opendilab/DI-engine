template.qmix
-------------------------------------------------------

Please Reference nerveX/nervex/docs/source/api_doc/model/template/qmix.py for usage

Mixer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.model.template.qmix.Mixer
    :members: __init__, forward

PMixer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.model.template.qmix.PMixer
    :members: __init__, forward


QMix
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.model.template.qmix.QMix
    :members: __init__, forward, _setup_global_encoder

CollaQMultiHeadAttention
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.model.template.qmix.CollaQMultiHeadAttention
    :members: __init__, forward

CollaQSMACAttentionModule
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.model.template.qmix.CollaQSMACAttentionModule
    :members: __init__, _cut_obs, forward


CollaQ
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.model.template.qmix.CollaQ
    :members: __init__, forward, _setup_global_encoder