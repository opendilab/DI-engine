rl_utils.exploration
======================

exploration
-----------------

epsilon_greedy
~~~~~~~~~~~~~~~~
.. automodule:: nervex.rl_utils.exploration.epsilon_greedy

BaseNoise
~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.rl_utils.exploration.BaseNoise
    :members: __init__, __call__

GaussianNoise
~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.rl_utils.exploration.GaussianNoise
    :members: __init__, __call__

OUNoise
~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.rl_utils.exploration.OUNoise
    :members: __init__, __call__, reset

create_noise_generator
~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.rl_utils.exploration.create_noise_generator
