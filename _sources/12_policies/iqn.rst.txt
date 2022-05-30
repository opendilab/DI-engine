IQN
^^^^^^^

Overview
---------
IQN was proposed in `Implicit Quantile Networks for Distributional Reinforcement Learning <https://arxiv.org/pdf/1806.06923>`_. The key difference between IQN and QRDQN is that IQN introduces the implicit quantile network (IQN), a deterministic parametric function trained to re-parameterize samples from a base distribution, e.g. tau in U([0, 1]), to the respective quantile values of a target distribution, while QRDQN direct learns a fixed set of pre-defined quantiles.

Quick Facts
-----------
1. IQN is a **model-free** and **value-based** RL algorithm.

2. IQN only support **discrete action spaces**.

3. IQN is an **off-policy** algorithm.

4. Usually, IQN use **eps-greedy** or **multinomial sample** for exploration.

5. IQN can be equipped with RNN.

Key Equations
-------------
In implicit quantile networks, a sampled quantile tau is first encoded into an embedding vector via:

    .. math::

        \phi_{j}(\tau):=\operatorname{ReLU}\left(\sum_{i=0}^{n-1} \cos (\pi i \tau) w_{i j}+b_{j}\right)

Then the quantile embedding is element-wise multiplied by the embedding of the observation of the environment, and the subsequent fully-connected layers map the resulted product vector to the respective quantile value.

Key Graphs
-------------
The comparison among DQN, C51, QRDQN and IQN is shown as follows:

.. image:: images/dis_reg_compare.png
   :align: center
   :width: 800

Extensions
-----------
IQN can be combined with:
  - PER (Prioritized Experience Replay)

    .. tip::
        Whether PER improves IQN depends on the task and the training strategy.
  - Multi-step TD-loss
  - Double (target) Network
  - RNN

Implementation
------------------

.. tip::
      Our benchmark result of IQN uses the same hyper-parameters as DQN except the IQN's exclusive hyper-parameter, ``the number of quantiles``, which is empirically set as 32. The number of quantiles are not recommended to set larger than 64, which brings marginal gain and much more forward latency.

The default config of IQN is defined as follows:

.. autoclass:: ding.policy.iqn.IQNPolicy
   :noindex:

The network interface IQN used is defined as follows:

.. autoclass:: ding.model.template.q_learning.IQN
   :members: forward
   :noindex:

The bellman updates of IQN used is defined in the function ``iqn_nstep_td_error`` of ``ding/rl_utils/td.py``.

Benchmark
-----------

+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| environment         |best mean reward | evaluation results                                  | config link              | comparison           |
+=====================+=================+=====================================================+==========================+======================+
|                     |                 |                                                     |`config_link_p <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Tianshou(20)        |
|                     |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|Pong                 |  20             |.. image:: images/benchmark/IQN_pong.png             |atari/config/serial/      |                      |
|                     |                 |                                                     |pong/pong_iqn_config      |                      |
|(PongNoFrameskip-v4) |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_q <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Tianshou(15520)     |
|Qbert                |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |  16331          |.. image:: images/benchmark/IQN_qbert.png            |atari/config/serial/      |                      |
|(QbertNoFrameskip-v4)|                 |                                                     |qbert/qbert_iqn_config    |                      |
|                     |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_s <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Tianshou(1370)      |
|SpaceInvaders        |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |  1493           |.. image:: images/benchmark/IQN_spaceinvaders.png    |atari/config/serial/      |                      |
|(SpaceInvadersNoFrame|                 |                                                     |spaceinvaders/space       |                      |
|skip-v4)             |                 |                                                     |invaders_iqn_config.py>`_ |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+

P.S.:
  1. The above results are obtained by running the same configuration on five different random seeds (0, 1, 2, 3, 4).

References
------------


(IQN) Will Dabney, Georg Ostrovski, David Silver, Rémi Munos: “Implicit Quantile Networks for Distributional Reinforcement Learning”, 2018; arXiv:1806.06923. https://arxiv.org/pdf/1806.06923


Other Public Implementations
---------------------------------

  - `Tianshou <https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/iqn.py>`_
