FQF
^^^^^^^

Overview
---------
FQF was proposed in `Fully Parameterized Quantile Function for Distributional Reinforcement Learning <https://arxiv.org/pdf/1911.02140>`_. The key difference between FQF and IQN is that FQF additionally introduces the fraction proposal network, a parametric function trained to generate tau in [0, 1], while IQN samples tau from a base distribution, e.g. U([0, 1]).

Quick Facts
-----------
1. FQF is a **model-free** and **value-based** distibutional RL algorithm.

2. FQF only support **discrete action spaces**.

3. FQF is an **off-policy** algorithm.

4. Usually, FQF use **eps-greedy** or **multinomial sample** for exploration.

5. FQF can be equipped with RNN.

Key Equations or Key Graphs
-------------
For any continuous quantile function :math:`F_{Z}^{-1}` that is non-decreasing, define the 1-Wasserstein loss of :math:`F_{Z}^{-1}` and :math:`F_{Z}^{-1, \tau}` by

.. math::

    W_{1}(Z, \tau)=\sum_{i=0}^{N-1} \int_{\tau_{i}}^{\tau_{i+1}}\left|F_{Z}^{-1}(\omega)-F_{Z}^{-1}\left(\hat{\tau}_{i}\right)\right| d \omega

Note that as :math:`W_{1}` is not computed, we can’t directly perform gradient descent for the fraction proposal network. Instead, we assign :math:`\frac{\partial W_{1}}{\partial \tau_{i}}` 
to the optimizer.

:math:`\frac{\partial W_{1}}{\partial \tau_{i}}` is given by

.. math::

    \frac{\partial W_{1}}{\partial \tau_{i}}=2 F_{Z}^{-1}\left(\tau_{i}\right)-F_{Z}^{-1}\left(\hat{\tau}_{i}\right)-F_{Z}^{-1}\left(\hat{\tau}_{i-1}\right), \forall i \in(0, N).

Like implicit quantile networks, a learned quantile tau is encoded into an embedding vector via:

.. math::

        \phi_{j}(\tau):=\operatorname{ReLU}\left(\sum_{i=0}^{n-1} \cos (\pi i \tau) w_{i j}+b_{j}\right)

Then the quantile embedding is element-wise multiplied by the embedding of the observation of the environment, and the subsequent fully-connected layers map the resulted product vector to the respective quantile value.

The advantage of FQF over IQN can be showed in this picture:

.. image:: images/fqf_iqn_compare.png
   :align: center
   :scale: 100%

Pseudo-code
-------------
.. image:: images/FQF.png
   :align: center
   :scale: 100%

Extensions
-----------
FQF can be combined with:

  - PER (Prioritized Experience Replay)

    .. tip::
        Whether PER improves FQF depends on the task and the training strategy.

  - Multi-step TD-loss
  - Double (target) Network
  - RNN

Implementation
------------------

.. tip::
      Our benchmark result of FQF uses the same hyper-parameters as DQN except the FQF's exclusive hyper-parameter, ``the number of quantiles``, which is empirically set as 32. Intuitively, the advantage of trained quantile fractions compared to random ones will be more observable at smaller N. At larger N when both trained quantile fractions and random ones are densely distributed over [0, 1], the differences between FQF and IQN becomes negligible.

The default config of FQF is defined as follows:

.. autoclass:: ding.policy.fqf.FQFPolicy
   :noindex:

The network interface FQF used is defined as follows:

.. autoclass:: ding.model.template.q_learning.FQF
   :members: forward
   :noindex:

The bellman updates of FQF used is defined in the function ``fqf_nstep_td_error`` of ``ding/rl_utils/td.py``.

Benchmark
-----------

+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| environment         |best mean reward | evaluation results                                  | config link              | comparison           |
+=====================+=================+=====================================================+==========================+======================+
|                     |                 |                                                     |`config_link_p <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Tianshou(20.7)      |
|                     |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|Pong                 |  21             |.. image:: images/benchmark/FQF_pong.png             |atari/config/serial/      |                      |
|                     |                 |                                                     |pong/pong_fqf_config      |                      |
|(PongNoFrameskip-v4) |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_q <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Tianshou(16172.5)   |
|Qbert                |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |  23416          |.. image:: images/benchmark/FQF_qbert.png            |atari/config/serial/      |                      |
|(QbertNoFrameskip-v4)|                 |                                                     |qbert/qbert_fqf_config    |                      |
|                     |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_s <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Tianshou(2482)      |
|SpaceInvaders        |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |  2727.5         |.. image:: images/benchmark/FQF_spaceinvaders.png    |atari/config/serial/      |                      |
|(SpaceInvadersNoFrame|                 |                                                     |spaceinvaders/space       |                      |
|skip-v4)             |                 |                                                     |invaders_fqf_config.py>`_ |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+

P.S.:
  1. The above results are obtained by running the same configuration on three different random seeds (0, 1, 2).

References
------------


(FQF) Derek Yang, Li Zhao, Zichuan Lin, Tao Qin, Jiang Bian, Tieyan Liu: “Fully Parameterized Quantile Function for Distributional Reinforcement Learning”, 2019; arXiv:1911.02140. https://arxiv.org/pdf/1911.02140


Other Public Implementations
---------------------------------

- `Tianshou <https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/fqf.py>`_
