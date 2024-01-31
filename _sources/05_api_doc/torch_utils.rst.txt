ding.torch_utils
------------------


loss
========
Please refer to ``ding/torch_utils/loss`` for more details.

ContrastiveLoss
~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.loss.ContrastiveLoss
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:


LabelSmoothCELoss
~~~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.loss.LabelSmoothCELoss
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

SoftFocalLoss
~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.loss.SoftFocalLoss
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:


build_ce_criterion
~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.loss.build_ce_criterion


MultiLogitsLoss
~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.loss.MultiLogitsLoss
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:


network.activation
==================
Please refer to ``ding/torch_utils/network/activation`` for more details.

Lambda
~~~~~~~
.. autoclass:: ding.torch_utils.network.activation.Lambda
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

GLU
~~~~~~~
.. autoclass:: ding.torch_utils.network.activation.GLU
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

Swish
~~~~~~~
.. autoclass:: ding.torch_utils.network.activation.Swish
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

GELU
~~~~~~~
.. autoclass:: ding.torch_utils.network.activation.GELU
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

build_activation
~~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.activation.build_activation


network.diffusion
=================
Please refer to ``ding/torch_utils/network/diffusion`` for more details.

extract
~~~~~~~
.. autofunction:: ding.torch_utils.network.diffusion.extract

cosine_beta_schedule
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.diffusion.cosine_beta_schedule

apply_conditioning
~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.diffusion.apply_conditioning

DiffusionConv1d
~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.diffusion.DiffusionConv1d
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

SinusoidalPosEmb
~~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.diffusion.SinusoidalPosEmb
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

Residual
~~~~~~~~
.. autoclass:: ding.torch_utils.network.diffusion.Residual
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

LayerNorm
~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.diffusion.LayerNorm
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

PreNorm
~~~~~~~
.. autoclass:: ding.torch_utils.network.diffusion.PreNorm
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

LinearAttention
~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.diffusion.LinearAttention
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

ResidualTemporalBlock
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.diffusion.ResidualTemporalBlock
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

DiffusionUNet1d
~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.diffusion.DiffusionUNet1d
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

TemporalValue
~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.diffusion.TemporalValue
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

network.dreamer
===============
Please refer to ``ding/torch_utils/network/dreamer`` for more details.

Conv2dSame
~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.dreamer.Conv2dSame
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

DreamerLayerNorm
~~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.dreamer.DreamerLayerNorm
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

DenseHead
~~~~~~~~~
.. autoclass:: ding.torch_utils.network.dreamer.DenseHead
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

ActionHead
~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.dreamer.ActionHead
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

SampleDist
~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.dreamer.SampleDist
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

OneHotDist
~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.dreamer.OneHotDist
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

TwoHotDistSymlog
~~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.dreamer.TwoHotDistSymlog
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

SymlogDist
~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.dreamer.SymlogDist
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

ContDist
~~~~~~~~
.. autoclass:: ding.torch_utils.network.dreamer.ContDist
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

Bernoulli
~~~~~~~~~
.. autoclass:: ding.torch_utils.network.dreamer.Bernoulli
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

network.gtrxl
=============
Please refer to ``ding/torch_utils/network/gtrxl`` for more details.

PositionalEmbedding
~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.gtrxl.PositionalEmbedding
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

GRUGatingUnit
~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.gtrxl.GRUGatingUnit
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

Memory
~~~~~~~
.. autoclass:: ding.torch_utils.network.gtrxl.Memory
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

AttentionXL
~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.gtrxl.AttentionXL
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

GatedTransformerXLLayer
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.gtrxl.GatedTransformerXLLayer
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

GTrXL
~~~~~~~
.. autoclass:: ding.torch_utils.network.gtrxl.GTrXL
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:


network.gumbel_softmax
======================
Please refer to ``ding/torch_utils/network/gumbel_softmax`` for more details.

GumbelSoftmax
~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.gumbel_softmax.GumbelSoftmax
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

network.merge
=============
Please refer to ``ding/torch_utils/network/merge`` for more details.

BilinearGeneral
~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.merge.BilinearGeneral
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

TorchBilinearCustomized
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.merge.TorchBilinearCustomized
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

FiLM
~~~~~~~
.. autoclass:: ding.torch_utils.network.merge.FiLM
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

GatingType
~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.merge.GatingType
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

SumMerge
~~~~~~~~
.. autoclass:: ding.torch_utils.network.merge.SumMerge
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

VectorMerge
~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.merge.VectorMerge
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:


network.nn_module
=================
Please refer to ``ding/torch_utils/network/nn_module`` for more details.

weight_init_
~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.nn_module.weight_init_

sequential_pack
~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.nn_module.sequential_pack

conv1d_block
~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.nn_module.conv1d_block

conv2d_block
~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.nn_module.conv2d_block

deconv2d_block
~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.nn_module.deconv2d_block

fc_block
~~~~~~~~
.. autofunction:: ding.torch_utils.network.nn_module.fc_block

normed_linear
~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.nn_module.normed_linear

normed_conv2d
~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.nn_module.normed_conv2d

MLP
~~~~~~~
.. autofunction:: ding.torch_utils.network.nn_module.MLP

ChannelShuffle
~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.nn_module.ChannelShuffle
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

one_hot
~~~~~~~
.. autofunction:: ding.torch_utils.network.nn_module.one_hot

NearestUpsample
~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.nn_module.NearestUpsample
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

BilinearUpsample
~~~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.nn_module.BilinearUpsample
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

binary_encode
~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.nn_module.binary_encode

NoiseLinearLayer
~~~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.nn_module.NoiseLinearLayer
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

noise_block
~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.nn_module.noise_block

NaiveFlatten
~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.nn_module.NaiveFlatten
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

network.normalization
=====================
Please refer to ``ding/torch_utils/network/normalization`` for more details.

build_normalization
~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.normalization.build_normalization

network.popart
==============
Please refer to ``ding/torch_utils/network/popart`` for more details.

PopArt
~~~~~~~
.. autoclass:: ding.torch_utils.network.popart.PopArt
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

network.res_block
=================
Please refer to ``ding/torch_utils/network/res_block`` for more details.

ResBlock
~~~~~~~~
.. autoclass:: ding.torch_utils.network.res_block.ResBlock
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

ResFCBlock
~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.res_block.ResFCBlock
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

network.resnet
==============
Please refer to ``ding/torch_utils/network/resnet`` for more details.

to_2tuple
~~~~~~~~~
.. autofunction:: ding.torch_utils.network.resnet.to_2tuple

get_same_padding
~~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.resnet.get_same_padding

pad_same
~~~~~~~~
.. autofunction:: ding.torch_utils.network.resnet.pad_same

avg_pool2d_same
~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.resnet.avg_pool2d_same

AvgPool2dSame
~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.resnet.AvgPool2dSame
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

create_classifier
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.resnet.create_classifier

ClassifierHead
~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.resnet.ClassifierHead
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

create_attn
~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.resnet.create_attn

get_padding
~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.resnet.get_padding

BasicBlock
~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.resnet.BasicBlock
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

Bottleneck
~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.resnet.Bottleneck
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

downsample_conv
~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.resnet.downsample_conv

downsample_avg
~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.resnet.downsample_avg

drop_blocks
~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.resnet.drop_blocks

make_blocks
~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.resnet.make_blocks

ResNet
~~~~~~~
.. autoclass:: ding.torch_utils.network.resnet.ResNet
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

resnet18
~~~~~~~~
.. autofunction:: ding.torch_utils.network.resnet.resnet18

network.rnn
===========
Please refer to ``ding/torch_utils/network/rnn`` for more details.

is_sequence
~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.rnn.is_sequence

sequence_mask
~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.rnn.sequence_mask

LSTMForwardWrapper
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.rnn.LSTMForwardWrapper
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

LSTM
~~~~~~~
.. autoclass:: ding.torch_utils.network.rnn.LSTM
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

PytorchLSTM
~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.rnn.PytorchLSTM
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

GRU
~~~~~~~
.. autoclass:: ding.torch_utils.network.rnn.GRU
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

get_lstm
~~~~~~~~~
.. autofunction:: ding.torch_utils.network.rnn.get_lstm


network.scatter_connection
==========================
Please refer to ``ding/torch_utils/network/scatter_connection`` for more details.

shape_fn_scatter_connection
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.network.scatter_connection.shape_fn_scatter_connection

ScatterConnection
~~~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.scatter_connection.ScatterConnection
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:


network.soft_argmax
===================
Please refer to ``ding/torch_utils/network/soft_argmax`` for more details.

SoftArgmax
~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.soft_argmax.SoftArgmax
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

network.transformer
===================
Please refer to ``ding/torch_utils/network/transformer`` for more details.

Attention
~~~~~~~~~
.. autoclass:: ding.torch_utils.network.transformer.Attention
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

TransformerLayer
~~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.transformer.TransformerLayer
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

Transformer
~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.transformer.Transformer
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

ScaledDotProductAttention
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.network.transformer.ScaledDotProductAttention
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

backend_helper
==============
Please refer to ``ding/torch_utils/backend_helper`` for more details.

enable_tf32
~~~~~~~~~~~
.. autofunction:: ding.torch_utils.backend_helper.enable_tf32

checkpoint_helper
=================
Please refer to ``ding/torch_utils/checkpoint_helper`` for more details.

build_checkpoint_helper
~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.checkpoint_helper.build_checkpoint_helper

CheckpointHelper
~~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.checkpoint_helper.CheckpointHelper
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

CountVar
~~~~~~~~
.. autoclass:: ding.torch_utils.checkpoint_helper.CountVar
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

auto_checkpoint
~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.checkpoint_helper.auto_checkpoint


data_helper
===========
Please refer to ``ding/torch_utils/data_helper`` for more details.

to_device
~~~~~~~~~
.. autofunction:: ding.torch_utils.data_helper.to_device

to_dtype
~~~~~~~~
.. autofunction:: ding.torch_utils.data_helper.to_dtype

to_tensor
~~~~~~~~~
.. autofunction:: ding.torch_utils.data_helper.to_tensor

to_ndarray
~~~~~~~~~~
.. autofunction:: ding.torch_utils.data_helper.to_ndarray

to_list
~~~~~~~
.. autofunction:: ding.torch_utils.data_helper.to_list

tensor_to_list
~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.data_helper.tensor_to_list

to_item
~~~~~~~
.. autofunction:: ding.torch_utils.data_helper.to_item

same_shape
~~~~~~~~~~
.. autofunction:: ding.torch_utils.data_helper.same_shape

LogDict
~~~~~~~
.. autoclass:: ding.torch_utils.data_helper.LogDict
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

build_log_buffer
~~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.data_helper.build_log_buffer

CudaFetcher
~~~~~~~~~~~
.. autoclass:: ding.torch_utils.data_helper.CudaFetcher
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

get_tensor_data
~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.data_helper.get_tensor_data

unsqueeze
~~~~~~~~~
.. autofunction:: ding.torch_utils.data_helper.unsqueeze

squeeze
~~~~~~~
.. autofunction:: ding.torch_utils.data_helper.squeeze

get_null_data
~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.data_helper.get_null_data

zeros_like
~~~~~~~~~~
.. autofunction:: ding.torch_utils.data_helper.zeros_like

dataparallel
============
Please refer to ``ding/torch_utils/dataparallel`` for more details.

DataParallel
~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.dataparallel.DataParallel
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

distribution
============
Please refer to ``ding/torch_utils/distribution`` for more details.

Pd
~~~~~~~
.. autoclass:: ding.torch_utils.distribution.Pd
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

CategoricalPd
~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.distribution.CategoricalPd
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

CategoricalPdPytorch
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.distribution.CategoricalPdPytorch
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

lr_scheduler
=============
Please refer to ``ding/torch_utils/lr_scheduler`` for more details.

get_lr_ratio
~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.lr_scheduler.get_lr_ratio

cos_lr_scheduler
~~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.lr_scheduler.cos_lr_scheduler


math_helper
===========
Please refer to ``ding/torch_utils/math_helper`` for more details.

cov
~~~~~~~
.. autofunction:: ding.torch_utils.math_helper.cov

metric
========
Please refer to ``ding/torch_utils/metric`` for more details.

levenshtein_distance
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.metric.levenshtein_distance

hamming_distance
~~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.metric.hamming_distance

model_helper
============
Please refer to ``ding/torch_utils/model_helper`` for more details.

get_num_params
~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.model_helper.get_num_params

nn_test_helper
==============
Please refer to ``ding/torch_utils/nn_test_helper`` for more details.

is_differentiable
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.nn_test_helper.is_differentiable

optimizer_helper
================
Please refer to ``ding/torch_utils/optimizer_helper`` for more details.

calculate_grad_norm
~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.optimizer_helper.calculate_grad_norm

calculate_grad_norm_without_bias_two_norm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.optimizer_helper.calculate_grad_norm_without_bias_two_norm

grad_ignore_norm
~~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.optimizer_helper.grad_ignore_norm

grad_ignore_value
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.optimizer_helper.grad_ignore_value

Adam
~~~~~~~
.. autoclass:: ding.torch_utils.optimizer_helper.Adam
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

RMSprop
~~~~~~~
.. autoclass:: ding.torch_utils.optimizer_helper.RMSprop
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

PCGrad
~~~~~~~
.. autoclass:: ding.torch_utils.optimizer_helper.PCGrad
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

configure_weight_decay
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.optimizer_helper.configure_weight_decay


parameter
=========
Please refer to ``ding/torch_utils/parameter`` for more details.

NonegativeParameter
~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.parameter.NonegativeParameter
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

TanhParameter
~~~~~~~~~~~~~
.. autoclass:: ding.torch_utils.parameter.TanhParameter
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

reshape_helper
==============
Please refer to ``ding/torch_utils/reshape_helper`` for more details.

fold_batch
~~~~~~~~~~~
.. autofunction:: ding.torch_utils.reshape_helper.fold_batch

unfold_batch
~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.reshape_helper.unfold_batch

unsqueeze_repeat
~~~~~~~~~~~~~~~~
.. autofunction:: ding.torch_utils.reshape_helper.unsqueeze_repeat
