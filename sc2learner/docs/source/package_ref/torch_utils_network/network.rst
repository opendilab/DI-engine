torch_utils.network
===================

nn_module
---------

weight_init_
~~~~~~~~~~~~~~~

.. automodule:: sc2learner.torch_utils.network.nn_module.weight_init_


sequential_pack
~~~~~~~~~~~~~~~

.. automodule:: sc2learner.torch_utils.network.nn_module.sequential_pack

conv1d_block
~~~~~~~~~~~~~~~

.. automodule:: sc2learner.torch_utils.network.nn_module.conv1d_block

conv2d_block
~~~~~~~~~~~~~~~

.. automodule:: sc2learner.torch_utils.network.nn_module.conv2d_block

deconv2d_block
~~~~~~~~~~~~~~~

.. automodule:: sc2learner.torch_utils.network.nn_module.deconv2d_block


fc_block
~~~~~~~~~~~~~~~

.. automodule:: sc2learner.torch_utils.network.nn_module.fc_block

one_hot
~~~~~~~~~~~~~~~

.. automodule:: sc2learner.torch_utils.network.nn_module.one_hot

binary_encode
~~~~~~~~~~~~~~~

.. automodule:: sc2learner.torch_utils.network.nn_module.binary_encode

BilinearUpsample
~~~~~~~~~~~~~~~~

.. autoclass:: sc2learner.torch_utils.network.nn_module.BilinearUpsample
    :members: __init__, forward

NearestUpsample
~~~~~~~~~~~~~~~~

.. autoclass:: sc2learner.torch_utils.network.nn_module.NearestUpsample
    :members: __init__, forward


ChannelShuffle
~~~~~~~~~~~~~~~~

.. autoclass:: sc2learner.torch_utils.network.nn_module.ChannelShuffle
    :members: __init__, forward




soft_argmax
------------



SoftArgmax
~~~~~~~~~~~~~~~~

.. autoclass:: sc2learner.torch_utils.network.soft_argmax.SoftArgmax
    :members: __init__, forward




activation
------------



GLU
~~~~~~~~~~~~~~~~

.. autoclass:: sc2learner.torch_utils.network.activation.GLU
    :members: __init__, forward



build_activation
~~~~~~~~~~~~~~~~~

.. automodule:: sc2learner.torch_utils.network.activation.build_activation




block
---------

ResBlock
~~~~~~~~~~~~~~~~

.. autoclass:: sc2learner.torch_utils.network.block.ResBlock
    :members: __init__, forward



ResFCBlock
~~~~~~~~~~~~~~~~

.. autoclass:: sc2learner.torch_utils.network.block.ResFCBlock
    :members: __init__, forward


normalization
-----------------

GroupSyncBatchNorm
~~~~~~~~~~~~~~~~~~~

.. autoclass:: sc2learner.torch_utils.network.normalization.GroupSyncBatchNorm
    :members: __init__, __repr__


AdaptiveInstanceNorm2d
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sc2learner.torch_utils.network.normalization.AdaptiveInstanceNorm2d
    :members: __init__, forward


build_normalization
~~~~~~~~~~~~~~~~~~~~

.. automodule:: sc2learner.torch_utils.network.normalization.build_normalization





