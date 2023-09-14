ding.model
----------

Common
========
Please refer to ``ding/model/common`` for more details.

create_model
~~~~~~~~~~~~~
.. autofunction:: ding.model.create_model

ConvEncoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.ConvEncoder
    :members: __init__, forward

FCEncoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.FCEncoder
    :members: __init__, forward


IMPALAConvEncoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.IMPALAConvEncoder
    :members: __init__

DiscreteHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.DiscreteHead
    :members: __init__, forward


DistributionHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.DistributionHead
    :members: __init__, forward


RainbowHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.RainbowHead
    :members: __init__, forward


QRDQNHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.QRDQNHead
    :members: __init__, forward

QuantileHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.QuantileHead
    :members: __init__, quantile_net, forward

FQFHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.FQFHead
    :members: __init__, quantile_net, forward

DuelingHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.DuelingHead
    :members: __init__, forward

StochasticDuelingHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.StochasticDuelingHead
    :members: __init__, forward

BranchingHead
~~~~~~~~~~~~~

.. autoclass:: ding.model.BranchingHead
    :members: __init__, forward

RegressionHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.RegressionHead
    :members: __init__, forward

ReparameterizationHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.ReparameterizationHead
    :members: __init__, forward

AttentionPolicyHead
~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.AttentionPolicyHead
    :members: __init__, forward

MultiHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.MultiHead
    :members: __init__, forward


independent_normal_dist
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ding.model.independent_normal_dist

Template
========
Please refer to ``ding/model/template`` for more details.


DQN
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.DQN
    :members: __init__, forward

C51DQN
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.C51DQN
    :members: __init__, forward


QRDQN
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.QRDQN
    :members: __init__, forward

IQN
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.IQN
    :members: __init__, forward

FQF
~~~~
.. autoclass:: ding.model.FQF
    :members: __init__, forward

BDQ
~~~~
.. autoclass:: ding.model.BDQ
    :members: __init__, forward

RainbowDQN
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.RainbowDQN
    :members: __init__, forward
    :noindex:


DRQN
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.DRQN
    :members: __init__, forward


GTrXLDQN
~~~~~~~~~~
.. autoclass:: ding.model.GTrXLDQN
    :members: __init__, forward, reset_memory, get_memory


VAC
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.VAC
    :members: __init__, forward, compute_actor, compute_critic, compute_actor_critic


ContinuousQAC
~~~~~~~~~~~~~~~
.. autoclass:: ding.model.ContinuousQAC
    :members: __init__, forward, compute_actor, compute_critic


DiscreteQAC
~~~~~~~~~~~
.. autoclass:: ding.model.DiscreteQAC
    :members: __init__, forward, compute_actor, compute_critic

QACDIST
~~~~~~~~~~~
.. autoclass:: ding.model.QACDIST
    :members: __init__, forward

DiscreteBC
~~~~~~~~~~~
.. autoclass:: ding.model.DiscreteBC
    :members: __init__, forward

ContinuousBC
~~~~~~~~~~~~~~
.. autoclass:: ding.model.ContinuousBC
    :members: __init__, forward

DREAMERVAC
~~~~~~~~~~~~~~
.. autoclass:: ding.model.DREAMERVAC

PDQN
~~~~
.. autoclass:: ding.model.PDQN
    :members: __init__, forward

DecisionTransformer
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.DecisionTransformer
    :members: __init__, forward

QMix
~~~~
.. autoclass:: ding.model.QMix
    :members: __init__, forward

PPG
~~~~
.. autoclass:: ding.model.PPG
    :members: __init__, forward

MAVAC
~~~~~~
.. autoclass:: ding.model.MAVAC
    :members: __init__, forward

DiscreteMAQAC
~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.DiscreteMAQAC
    :members: __init__, forward

ContinuousMAQAC
~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.ContinuousMAQAC
    :members: __init__, forward


ProcedureCloningBFS
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.ProcedureCloningBFS
    :members: __init__, forward

ProcedureCloningMCTS
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.ProcedureCloningMCTS
    :members: __init__, forward


BCQ
~~~~
.. autoclass:: ding.model.BCQ
    :members: __init__, forward


EDAC
~~~~
.. autoclass:: ding.model.EDAC
    :members: __init__, forward

Wrapper
=======
Please refer to ``ding/model/wrapper`` for more details.

IModelWrapper
~~~~~~~~~~~~~~
.. autoclass:: ding.model.IModelWrapper
    :members: __init__, __getattr__, info, reset, forward

model_wrap
~~~~~~~~~~~~
.. autofunction:: ding.model.model_wrap

register_wrapper
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.model.register_wrapper

BaseModelWrapper
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.wrapper.model_wrappers.BaseModelWrapper
    :members: reset, forward

ArgmaxSampleWrapper
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.wrapper.model_wrappers.ArgmaxSampleWrapper
    :members: forward

MultinomialSampleWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.wrapper.model_wrappers.MultinomialSampleWrapper
    :members: forward

EpsGreedySampleWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.wrapper.model_wrappers.EpsGreedySampleWrapper
    :members: forward

EpsGreedyMultinomialSampleWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.wrapper.model_wrappers.EpsGreedyMultinomialSampleWrapper
    :members: forward

DeterministicSampleWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.wrapper.model_wrappers.DeterministicSampleWrapper
    :members: forward

ReparamSampleWrapper
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.wrapper.model_wrappers.ReparamSampleWrapper
    :members: forward



CombinationArgmaxSampleWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.wrapper.model_wrappers.CombinationArgmaxSampleWrapper
    :members: forward

CombinationMultinomialSampleWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.wrapper.model_wrappers.CombinationMultinomialSampleWrapper
    :members: forward

HybridArgmaxSampleWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.wrapper.model_wrappers.HybridArgmaxSampleWrapper
    :members: forward

HybridEpsGreedySampleWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.wrapper.model_wrappers.HybridEpsGreedySampleWrapper
    :members: forward

HybridEpsGreedyMultinomialSampleWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.wrapper.model_wrappers.HybridEpsGreedyMultinomialSampleWrapper
    :members: forward

HybridReparamMultinomialSampleWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.wrapper.model_wrappers.HybridReparamMultinomialSampleWrapper
    :members: forward

HybridDeterministicArgmaxSampleWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.wrapper.model_wrappers.HybridDeterministicArgmaxSampleWrapper
    :members: forward

ActionNoiseWrapper
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.wrapper.model_wrappers.ActionNoiseWrapper
    :members: __init__, forward

TargetNetworkWrapper
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.wrapper.model_wrappers.TargetNetworkWrapper
    :members: __init__, forward

HiddenStateWrapper
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.wrapper.model_wrappers.HiddenStateWrapper
    :members: __init__, reset, forward


TransformerInputWrapper
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.wrapper.model_wrappers.TransformerInputWrapper
    :members: __init__, reset, forward


TransformerSegmentWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.wrapper.model_wrappers.TransformerSegmentWrapper
    :members: __init__, forward

TransformerMemoryWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.model.wrapper.model_wrappers.TransformerMemoryWrapper
    :members: __init__, forward, reset
