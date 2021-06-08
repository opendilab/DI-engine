Soft Actor-Critic
^^^^^^^

Overview
---------
Soft actor-critic (SAC) is an off-policy maximum entropy actor-critic algorithm, which is proposed in the 2018 paper `Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor <https://arxiv.org/abs/1801.01290>`_

Quick Facts
-----------
1. SAC is implemented for environments with **continuous action spaces**.(i.e. MuJoCo)

2. SAC is an **off-policy** and **model-free** algorithm.

3. SAC is a **actor-critic** RL algorithm, which optimizes actor network and critic network, respectively,

Key Equations or Key Graphs
---------------------------
The entropy used in SAC is:

.. image:: images/entropy.png


Pseudocode
----------

.. image:: images/SAC.jpg

.. note::
   Compared with the vanilla version, we only optimize q network and actor network.


Extensions
-----------
SAC can be combined with:
    - Double soft Q Network
    - Replay Buffers
    - Gaussian noise during collecting transition

Implementation
---------------------------------
The default config is defined as follows:

.. autoclass:: nervex.policy.sac.SACPolicy

The Benchmark result of SAC implemented in nerveX is shown in `Benchmark <../feature/algorithm_overview.html>`_

Entropy-Regularized Reinforcement Learning
---------------------------------

Entropy in target q value
.. code-block:: python
     # the value of a policy according to the maximum entropy objective
    if self._twin_q:
        # find min one as target q value
        target_q_value = torch.min(target_q_value[0],
                                   target_q_value[1]) - self._alpha * next_data['log_prob'].squeeze(-1)
    else:
        target_q_value = target_q_value - self._alpha * next_data['log_prob'].squeeze(-1)

Entropy in policy
.. code-block:: python
    # compute policy loss
    if not self._reparameterization:
        target_log_policy = new_q_value - v_value
        policy_loss = (log_prob * (log_prob - target_log_policy.unsqueeze(-1))).mean()
    else:
        policy_loss = (self._alpha * log_prob - new_q_value.unsqueeze(-1)).mean()


Other Public Implementations
----------------------------

- Baselines_
- rllab_
- `rllib (Ray)`_
- Spinningup_
- tianshou_

.. _Baselines: https://github.com/openai/baselines/tree/master/baselines/sac
.. _rllab: https://github.com/rll/rllab/blob/master/rllab/algos/sac.py
.. _`rllib (Ray)`: https://github.com/ray-project/ray/tree/master/python/ray/rllib/agents/sac
.. _Spinningup: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac
.. _tianshou: https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/sac.py
