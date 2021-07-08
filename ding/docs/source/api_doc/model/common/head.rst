common.head
-------------------------------------------------------

Please Reference ding/ding/docs/source/api_doc/model/common/head.py for usage



.. code-block:: python

    head_cls_map = {
        # discrete
        'discrete': DiscreteHead,
        'dueling': DuelingHead,
        'distribution': DistributionHead,
        'rainbow': RainbowHead,
        'qrdqn': QRDQNHead,
        'quantile': QuantileHead,
        # continuous
        'regression': RegressionHead,
        'reparameterization': ReparameterizationHead,
        # multi
        'multi': MultiHead,
    }


   


DiscreteHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.common.head.DiscreteHead
    :members: __init__, forward


DistributionHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.common.head.DistributionHead
    :members: __init__, forward


RainbowHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.common.head.RainbowHead
    :members: __init__, forward


QRDQNHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.common.head.QRDQNHead
    :members: __init__, forward

QuantileHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.common.head.QuantileHead
    :members: __init__, quantile_net, forward

DuelingHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.common.head.DuelingHead
    :members: __init__, forward

RegressionHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.common.head.RegressionHead
    :members: __init__, forward

ReparameterizationHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.common.head.ReparameterizationHead
    :members: __init__, forward

MultiHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.model.common.head.MultiHead
    :members: __init__, forward

