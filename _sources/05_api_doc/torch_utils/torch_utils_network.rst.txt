network.activation
--------------------

GLU
~~~~~~~~~~~~~~~~

.. autoclass:: ding.torch_utils.network.activation.GLU
    :members: forward

build_activation
~~~~~~~~~~~~~~~~~

.. automodule:: ding.torch_utils.network.activation.build_activation





network.nn_module
--------------------

weight_init
~~~~~~~~~~~~~~~

.. automodule:: ding.torch_utils.network.nn_module.weight_init_

sequential_pack
~~~~~~~~~~~~~~~

.. automodule:: ding.torch_utils.network.nn_module.sequential_pack

conv1d_block
~~~~~~~~~~~~~~~

.. automodule:: ding.torch_utils.network.nn_module.conv1d_block

conv2d_block
~~~~~~~~~~~~~~~

.. automodule:: ding.torch_utils.network.nn_module.conv2d_block

deconv2d_block
~~~~~~~~~~~~~~~

.. automodule:: ding.torch_utils.network.nn_module.deconv2d_block

fc_block
~~~~~~~~~~~~~~~

.. automodule:: ding.torch_utils.network.nn_module.fc_block

MLP
~~~~~~~~~~~~~~~

.. automodule:: ding.torch_utils.network.nn_module.MLP

one_hot
~~~~~~~~~~~~~~~

.. automodule:: ding.torch_utils.network.nn_module.one_hot

binary_encode
~~~~~~~~~~~~~~~

.. automodule:: ding.torch_utils.network.nn_module.binary_encode

noise_block
~~~~~~~~~~~~~~~

.. automodule:: ding.torch_utils.network.nn_module.noise_block

ChannelShuffle
~~~~~~~~~~~~~~~~

.. autoclass:: ding.torch_utils.network.nn_module.ChannelShuffle
    :members: forward

NearestUpsample
~~~~~~~~~~~~~~~~

.. autoclass:: ding.torch_utils.network.nn_module.NearestUpsample
    :members: forward

BilinearUpsample
~~~~~~~~~~~~~~~~

.. autoclass:: ding.torch_utils.network.nn_module.BilinearUpsample
    :members: forward

NoiseLinearLayer
~~~~~~~~~~~~~~~~

.. autoclass:: ding.torch_utils.network.nn_module.NoiseLinearLayer
    :members: reset_noise, reset_parameters, forward




network.normalization
--------------------------


build_normalization
~~~~~~~~~~~~~~~~~~~~

.. automodule:: ding.torch_utils.network.normalization.build_normalization



network.res_block
--------------------

ResBlock
~~~~~~~~~~~~~~~~

.. autoclass:: ding.torch_utils.network.res_block.ResBlock
    :members: forward

ResFCBlock
~~~~~~~~~~~~~~~~

.. autoclass:: ding.torch_utils.network.res_block.ResFCBlock
    :members: forward




network.rnn
-----------------

LSTMForwardWrapper
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.torch_utils.network.rnn.LSTMForwardWrapper
    :members: _before_forward, _after_forward

LSTM
~~~~~~~~~~~~~~~~

.. autoclass:: ding.torch_utils.network.rnn.LSTM
    :members: forward

PytorchLSTM
~~~~~~~~~~~~~~

.. autoclass:: ding.torch_utils.network.rnn.PytorchLSTM
    :members: forward

get_lstm
~~~~~~~~~~~~~~~~~

.. automodule:: ding.torch_utils.network.rnn.get_lstm



network.scatter_connection
-------------------------------

ScatterConnection
~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.torch_utils.network.scatter_connection.ScatterConnection
    :members: forward




network.soft_argmax
---------------------

SoftArgmax
~~~~~~~~~~~~~~~~

.. autoclass:: ding.torch_utils.network.soft_argmax.SoftArgmax
    :members: forward





network.transformer
----------------------

Attention
~~~~~~~~~~~~

.. autoclass:: ding.torch_utils.network.transformer.Attention
    :members: forward, split

TransformerLayer
~~~~~~~~~~~~~~~~~

.. autoclass:: ding.torch_utils.network.transformer.TransformerLayer
    :members: forward

Transformer
~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.torch_utils.network.transformer.Transformer
    :members: forward








