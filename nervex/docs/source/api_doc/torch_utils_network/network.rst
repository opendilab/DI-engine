torch_utils.network
===================

nn_module
---------

weight_init_
~~~~~~~~~~~~~~~

.. automodule:: nervex.torch_utils.network.nn_module.weight_init_


sequential_pack
~~~~~~~~~~~~~~~

.. automodule:: nervex.torch_utils.network.nn_module.sequential_pack

conv1d_block
~~~~~~~~~~~~~~~

.. automodule:: nervex.torch_utils.network.nn_module.conv1d_block

conv2d_block
~~~~~~~~~~~~~~~

.. automodule:: nervex.torch_utils.network.nn_module.conv2d_block

deconv2d_block
~~~~~~~~~~~~~~~

.. automodule:: nervex.torch_utils.network.nn_module.deconv2d_block


fc_block
~~~~~~~~~~~~~~~

.. automodule:: nervex.torch_utils.network.nn_module.fc_block

one_hot
~~~~~~~~~~~~~~~

.. automodule:: nervex.torch_utils.network.nn_module.one_hot

binary_encode
~~~~~~~~~~~~~~~

.. automodule:: nervex.torch_utils.network.nn_module.binary_encode

BilinearUpsample
~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.nn_module.BilinearUpsample
    :members: __init__, forward

NearestUpsample
~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.nn_module.NearestUpsample
    :members: __init__, forward


ChannelShuffle
~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.nn_module.ChannelShuffle
    :members: __init__, forward




soft_argmax
------------



SoftArgmax
~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.soft_argmax.SoftArgmax
    :members: __init__, forward




activation
------------



GLU
~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.activation.GLU
    :members: __init__, forward



build_activation
~~~~~~~~~~~~~~~~~

.. automodule:: nervex.torch_utils.network.activation.build_activation




Resblock
---------

ResBlock
~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.res_block.ResBlock
    :members: __init__, forward



ResFCBlock
~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.res_block.ResFCBlock
    :members: __init__, forward


Transformer
-----------------

Transformer
~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.transformer.Transformer
    :members: __init__, forward

TransformerLayer
~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.torch_utils.network.transformer.TransformerLayer
    :members: __init__, forward

Attention
~~~~~~~~~~~~
.. autoclass:: nervex.torch_utils.network.transformer.Attention
    :members: __init__, forward, split


scatter_connection
---------------------

ScatterConnection
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.torch_utils.network.scatter_connection.ScatterConnection
    :members: __init__, forward


normalization
-----------------

GroupSyncBatchNorm
~~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.normalization.GroupSyncBatchNorm
    :members: __init__, __repr__



build_normalization
~~~~~~~~~~~~~~~~~~~~

.. automodule:: nervex.torch_utils.network.normalization.build_normalization



rnn
-----------------

LSTMForwardWrapper
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.rnn.LSTMForwardWrapper
    :members: _before_forward, _after_forward

LSTM
~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.rnn.LSTM
    :members: __init__, forward


PytorchLSTM
~~~~~~~~~~~~~~
.. autoclass:: nervex.torch_utils.network.rnn.PytorchLSTM
    :members: forward

get_lstm
~~~~~~~~~~~~~~~~~

.. automodule:: nervex.torch_utils.network.rnn.get_lstm
