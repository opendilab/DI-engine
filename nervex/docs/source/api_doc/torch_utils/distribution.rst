torch_utils.distribution
================================


Pd
~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.torch_utils.distribution.Pd
    :members: neglogp, entropy, noise_mode, mode, sample

CategoricalPd
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.torch_utils.distribution.CategoricalPd
    :members: update_logits, neglogp, entropy, noise_mode, mode, sample


CategoricalPdPytorch
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.torch_utils.distribution.CategoricalPdPytorch
    :members: update_logits, updata_probs, sample, neglogp, mode, entropy
