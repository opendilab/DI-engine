Twin Delayed DDPG
^^^^^^^

Overview
---------

Twin Delayed DDPG (TD3), proposed in the 2018 paper `Addressing Function Approximation Error in Actor-Critic Methods <https://arxiv.org/abs/1802.09477>`_, is an algorithm which learns a Q-function and a policy simultaneously.
DDPG is an actor-critic, model-free algorithm based on the deep deterministic policy gradient(DDPG) that can address overestimation bias and the accumulation of error in temporal difference methods in continuous action spaces.

Quick Facts
-----------
1. TD3 is only used for environments with **continuous action spaces**.(i.e. MuJoCo)

2. TD3 is an **off-policy** algorithm.

3. TD3 is a **model-free** and **actor-critic** RL algorithm, which optimizes actor network and critic network, respectively.

Key Equations or Key Graphs
---------------------------
The target update of our Clipped Double Q-learning algorithm:

.. math::
    y_{1}=r+\gamma \min _{i=1,2} Q_{\theta_{i}^{\prime}}\left(s^{\prime}, \pi_{\phi_{1}}\left(s^{\prime}\right)\right)

In implementation, computational costs can be reduced by using a single actor optimized with respect to :math:`Q_{\theta_1}` . We then use the same target :math:`y_2= y_1for Q_{\theta_2}`.

Pseudocode
----------

.. image:: images/td3.jpg

Extensions
-----------
TD3 can be combined with:
    - Target Network and Target Policy Smoothing.
    - Policy Updates Delay.
    - Clipped Double-Q Learning.
    - Replay Buffers.
    - Gaussian noise during collecting transition.


Model
---------------------------------
Here we provide examples of `td3` model as default model for `DDPG`.


Implementations
----------------
The default config is defined as follows:

.. autoclass:: nervex.policy.ddpg.TD3Policy

The Benchmark result of TD3 implemented in nerveX is shown in `Benchmark <../feature/algorithm_overview.html>`_

Other Public Implementations
----------------------------

- Baselines_
- rllab_
- `rllib (Ray)`_
- `TD3 release repo`_
- Spinningup_
- tianshou_

.. _Baselines: https://github.com/openai/baselines/tree/master/baselines/ddpg
.. _rllab: https://github.com/rll/rllab/blob/master/rllab/algos/ddpg.py
.. _`rllib (Ray)`: https://github.com/ray-project/ray/tree/master/python/ray/rllib/agents/ddpg
.. _`TD3 release repo`: https://github.com/sfujim/TD3
.. _Spinningup: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ddpg
.. _tianshou: https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/ddpg.py
