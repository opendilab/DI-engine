network.activation
--------------------

GLU
~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.activation.GLU
    :members: forward

build_activation
~~~~~~~~~~~~~~~~~

.. automodule:: nervex.torch_utils.network.activation.build_activation





network.nn_module
--------------------

weight_init
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

MLP
~~~~~~~~~~~~~~~

.. automodule:: nervex.torch_utils.network.nn_module.MLP

one_hot
~~~~~~~~~~~~~~~

.. automodule:: nervex.torch_utils.network.nn_module.one_hot

binary_encode
~~~~~~~~~~~~~~~

.. automodule:: nervex.torch_utils.network.nn_module.binary_encode

noise_block
~~~~~~~~~~~~~~~

.. automodule:: nervex.torch_utils.network.nn_module.noise_block

ChannelShuffle
~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.nn_module.ChannelShuffle
    :members: forward

NearestUpsample
~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.nn_module.NearestUpsample
    :members: forward

BilinearUpsample
~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.nn_module.BilinearUpsample
    :members: forward

NoiseLinearLayer
~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.nn_module.NoiseLinearLayer
    :members: reset_noise, reset_parameters, forward




network.normalization
--------------------------

GroupSyncBatchNorm
~~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.normalization.GroupSyncBatchNorm
    :members: __init__

build_normalization
~~~~~~~~~~~~~~~~~~~~

.. automodule:: nervex.torch_utils.network.normalization.build_normalization



network.res_block
--------------------

ResBlock
~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.res_block.ResBlock
    :members: forward

ResFCBlock
~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.res_block.ResFCBlock
    :members: forward




network.rnn
-----------------

LSTMForwardWrapper
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.rnn.LSTMForwardWrapper
    :members: _before_forward, _after_forward

LSTM
~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.rnn.LSTM
    :members: forward

PytorchLSTM
~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.rnn.PytorchLSTM
    :members: forward

get_lstm
~~~~~~~~~~~~~~~~~

.. automodule:: nervex.torch_utils.network.rnn.get_lstm



network.scatter_connection
-------------------------------

ScatterConnection
~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.scatter_connection.ScatterConnection
    :members: forward




network.soft_argmax
---------------------

SoftArgmax
~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.soft_argmax.SoftArgmax
    :members: forward





network.transformer
----------------------

Attention
~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.transformer.Attention
    :members: forward, split

TransformerLayer
~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.transformer.TransformerLayer
    :members: forward

Transformer
~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.torch_utils.network.transformer.Transformer
    :members: forward








