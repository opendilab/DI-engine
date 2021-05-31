Key Concept
===============================

.. toctree::
   :maxdepth: 3

nerveX is an AI decision training framework, especially in deep reinforcement learning. It supports most traditional RL algorithms,
such as DQN, PPO, SAC and domain-specific algorithms like QMIX in multi-agent RL, GAIL in inverse RL and RND in exploration problems.
The whole supported algorithms introduction can be found in `link1 <>`.



Here we show some key concepts about reinforcement learning train and evaluate pipeline designed by nerveX.

Component
----------
Environment and policy is the most two important concepts in the total train program, in most cases, the users of nerveX only need to pay
attention to these two components, which are partially extended from the original definition in other RL papers and frameworks.

Env
~~~~~~~~

Policy
~~~~~~~

Config
~~~~~~~~~

Worker
~~~~~~~~~~~

If you want to know more details about algorithm implementation, framework design and efficiency optimization, we also provide the documation of ``Feature`` parts `<link2 <>`, 
please refer to it.

Algorithm Table
~~~~~~~~~~~~~~~~

Compute Pattern
-----------------

Serial Pipeline
~~~~~~~~~~~~~~~~~

Parallel/Dist Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~
TBD
