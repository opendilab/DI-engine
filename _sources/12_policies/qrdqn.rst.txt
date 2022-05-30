QRDQN
^^^^^^^

Overview
---------
QR (Quantile Regression) DQN was proposed in `Distributional Reinforcement Learning with Quantile Regression <https://arxiv.org/pdf/1710.10044>`_ and inherits the idea of learning the distribution of a q-value. Instead of approximate the distribution density function with discrete atoms, QRDQN, directly regresses a **discrete set of quantiles** of a q-value. 


Quick Facts
-----------
1. QRDQN is a **model-free** and **value-based** RL algorithm.

2. QRDQN only support **discrete action spaces**.

3. QRDQN is an **off-policy** algorithm.

4. Usually, QRDQN use **eps-greedy** or **multinomial sample** for exploration.

5. QRDQN can be equipped with RNN.

Key Equations or Key Graphs
----------------------------
C51 uses N fixed locations for its approximation distribution and adjusts their probabilities, while QRDQN assigns fixed, uniform probabilities to N adjustable locations. Based on this, QRDQN uses quantile regression to stochastically adjust the distributions’ locations so as to minimize
the Wasserstein distance to a target distribution.

The quantile regression loss, for a quantile :math:`\tau \in [0, 1]`, is an asymmetric convex loss function that penalizes **overestimation errors** with weight :math:`\tau` and **underestimation errors** with weight :math:`1−\tau`. For a distribution :math:`Z`, and a given quantile :math:`\tau`, the value of the quantile function :math:`F_Z^{−1}(\tau)` may be characterized as the minimizer of the quantile regression loss:

.. math::

   \begin{array}{r}
   \mathcal{L}_{\mathrm{QR}}^{\tau}(\theta):=\mathbb{E}_{\hat{z} \sim Z}\left[\rho_{\tau}(\hat{Z}-\theta)\right], \text { where } \\
   \rho_{\tau}(u)=u\left(\tau-\delta_{\{u<0\}}\right), \forall u \in \mathbb{R}
   \end{array}

And the above mentioned loss is not smooth at zero, which can limit performance when using non-linear function approximation. Therefore, a modified quantile loss, called ``quantile huber loss`` is applied during the Bellman update of QRDQN (i.e. the equation 10 in the following pseudo-code).

.. math::

   \rho^{\kappa}_{\tau}(u)=L_{\kappa}(u)\lvert \tau-\delta_{\{u<0\}} \rvert

Where :math:`L_{\kappa}` is Huber Loss.

.. note::

   Compared with DQN, QRDQN has these differences:

     1. Neural network architecture, the output layer of QRDQN is of size M x N, where M is the size of discrete action space and N is a hyper-parameter giving the number of quantile targets.
     2. Replace DQN loss with the quantile huber loss.
     3. In original QRDQN paper, replace RMSProp optimizer with Adam. While in DI-engine, we always use Adam optimizer.

Pseudo-code
-------------
.. image:: images/QRDQN.png
   :align: center
   :scale: 25%

Extensions
-----------
QRDQN can be combined with:

  - PER (Prioritized Experience Replay)
  - Multi-step TD-loss
  - Double (target) network
  - RNN

Implementation
----------------

.. tip::
      Our benchmark result of QRDQN uses the same hyper-parameters as DQN except the QRDQN's exclusive hyper-parameter, `the number of quantiles`, which is empirically set as 32.

The default config of QRDQN is defined as follows:

.. autoclass:: ding.policy.qrdqn.QRDQNPolicy
   :noindex:

The network interface QRDQN used is defined as follows:

.. autoclass:: ding.model.template.q_learning.QRDQN
   :members: forward
   :noindex:

The bellman updates of QRDQN is implemented in the function ``qrdqn_nstep_td_error`` of ``ding/rl_utils/td.py``.

Benchmark
------------

.. list-table:: Benchmark and comparison of QRDQN algorithm
   :widths: 25 15 30 15 15
   :header-rows: 1

   * - environment
     - best mean reward
     - evaluation results
     - config link
     - comparison
   * - | Pong
       | (PongNoFrameskip-v4)
     - 20
     - .. image:: images/benchmark/qrdqn_pong.png
     - `config_link_p <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/pong/pong_qrdqn_config.py>`_
     - | Tianshou (20)
   * - | Qbert
       | (QbertNoFrameskip-v4)
     - 18306
     - .. image:: images/benchmark/qrdqn_qbert.png
     - `config_link_q <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/qbert/qbert_qrdqn_config.py>`_
     - | Tianshou (14990)
   * - | SpaceInvaders
       | (SpaceInvadersNoFrame skip-v4)
     - 2231
     - .. image:: images/benchmark/qrdqn_spaceinvaders.png
     - `config_link_s <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/spaceinvaders/spaceinvaders_qrdqn_config.py>`_
     - | Tianshou (938)

P.S.：

1. The above results are obtained by running the same configuration on five different random seeds (0, 1, 2, 3, 4)
2. For the discrete action space algorithm like QRDQN, the Atari environment set is generally used for testing (including sub-environments Pong), and Atari environment is generally evaluated by the highest mean reward training 10M ``env_step``. For more details about Atari, please refer to `Atari Env Tutorial <../env_tutorial/atari.html>`_ .

References
------------

(QRDQN) Will Dabney, Mark Rowland, Marc G. Bellemare, Rémi Munos: “Distributional Reinforcement Learning with Quantile Regression”, 2017; arXiv:1710.10044. https://arxiv.org/pdf/1710.10044


Other Public Implementations
-------------------------------

- `Tianshou <https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/qrdqn.py>`_
