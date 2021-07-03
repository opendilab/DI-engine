common.head
-------------------------------------------------------

Please Reference nerveX/nervex/docs/source/api_doc/model/common/head.py for usage



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

.. autoclass:: nervex.model.common.head.DiscreteHead
    :members: __init__, forward


DistributionHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.model.common.head.DistributionHead
    :members: __init__, forward


RainbowHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.model.common.head.RainbowHead
    :members: __init__, forward


QRDQNHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.model.common.head.QRDQNHead
    :members: __init__, forward

QuantileHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.model.common.head.QuantileHead
    :members: __init__, quantile_net, forward

DuelingHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.model.common.head.DuelingHead
    :members: __init__, forward

RegressionHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.model.common.head.RegressionHead
    :members: __init__, forward

ReparameterizationHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.model.common.head.ReparameterizationHead
    :members: __init__, forward

MultiHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.model.common.head.MultiHead
    :members: __init__, forward

