DQN
^^^^^^^

Overview
---------
DQN was first proposed in `Playing Atari with Deep Reinforcement Learning <https://arxiv.org/abs/1312.5602>`_. Traditional Q-learning maintains an \ ``M*N`` \ Q value table (where M represents the number of states and N represents the number of actions), and iteratively updates the Q-value through the Bellman equation. This kind of algorithm will have the problem of dimensionality disaster when the state/action space becomes extremely large. 

DQN is different from traditional reinforcement learning methods. It combines Q-learning with deep neural networks, uses deep neural networks to estimate the Q value, and calculates the temporal-difference loss, and uses the gradient descent algorithm to update, thus reaching a level comparable to or even surpassing human players in problem decision-making in high-dimensional spaces (such as Atari games).

Quick Facts
-------------
1. DQN is a **model-free** and **value-based** RL algorithm.

2. DQN only support **discrete** action spaces.

3. DQN is an **off-policy** algorithm.

4. Usually, DQN uses **eps-greedy** or **multinomial sampling** for exploration.

5. DQN + RNN = DRQN.

6. The DI-engine implementation of DQN supports **multi-discrete** action space.

Key Equations or Key Graphs
---------------------------
The TD-loss used in DQN is:

.. math::

   L(w)=\mathbb{E}\left[(\underbrace{r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime}, w\right)}_{\text {Target }}-Q(s, a, w))^{2}\right]

Pseudo-code
---------------
.. image:: images/DQN.png
   :align: center
   :scale: 55%

The equation (3) in above pseudo-code is:

.. image:: images/eq3.png
   :align: center
   :scale: 40%

.. note::
   Compared with the vanilla version, DQN has been dramatically modified in both algorithm and implementation aspects. In the algorithm parts, **n-step TD-loss, PER, target network** and **dueling head** are widely used, interested users can refer to the paper `Rainbow: Combining Improvements in Deep Reinforcement Learning <https://arxiv.org/abs/1710.02298>`_ . For the implementation details, the value of epsilon anneals from a high value (e.g.: 0.95) to a low value (0.05) during the training rather than keeps constant, according to env steps (the number of policy interaction with env).

Extensions
-----------
DQN can be combined with:

    - PER (`Prioritized Experience Replay <https://arxiv.org/abs/1511.05952>`_)

      PER replaces the uniform sampling in a replay buffer with so-called ``priority`` defined by various metrics, such as absolute TD error, the novelty of observation and so on. By this priority sampling, the convergence speed and performance of DQN can be improved a lot.

      There are two kinds of implementation of PER. One of them is described below:

      .. image:: images/PERDQN.png
         :align: center
         :scale: 55%

      In DI-engine, PER can be enabled by modifying two fields ``priority`` and ``priority_IS_weight`` in the configuration file, and the concrete code can refer to `PER code <https://github.com/opendilab/DI-engine/blob/dev-treetensor/ding/worker/replay_buffer/advanced_buffer.py>`_ . For the specific example, users can refer to `PER example <../best_practice/priority.html>`_

    - Multi-step TD-loss
 
      In Single-step TD-loss, the update of Q-learning (Bellman equation) is described as:

        .. math::

          r(s,a)+\gamma \mathop{max}\limits_{a^*}Q(s',a^*)

      While in Multi-step TD-loss, it is overwritten by the following content:

        .. math::
           \sum_{t=0}^{n-1}\gamma^t r(s_t,a_t) + \gamma^n \mathop{max}\limits_{a^*}Q(s_n,a^*)

      .. note::
         An issue about n-step for Q-learning is that, when epsilon greedy is adopted, the q value estimation is biased because the :math:`r(s_t,a_t)` at t>=1 are sampled under epsilon greedy rather than the policy itself. However, multi-step along with epsilon greedy generally improves DQN practically.

      In DI-engine, Multi-step TD-loss can be enabled by the ``nstep`` field in the configuration file, and the loss function is described in ``q_nstep_td_error`` in `nstep code <https://github.com/opendilab/DI-engine/blob/dev-treetensor/ding/rl_utils/td.py>`_.

    - Double DQN (target network)

      Double DQN, proposed in `Deep Reinforcement Learning with Double Q-learning <https://arxiv.org/abs/1509.06461>`_, is a common variant of DQN. This method maintains another Q-network, named target network, which is updated by the current network by a fixed frequency (update times/training iterations).

      And the target Q in Double DQN is:

        .. image:: images/doubleDQN.png
           :align: center
           :scale: 20%

      Double DQN doesn't select the maximum q_value in the total discrete action space from the current network, but **first finds the action whose q_value is highest in the current network, then gets the q_value from the target network according to this selected action**. This variant can surpass the overestimation problem of target q_value, and reduce upward bias.

      Double DQN can suppress the over-estimation of Q value to reduce related negative impact.

      .. note::
            The overestimation can be caused by the error of function approximation(neural network for q table), environment noise, numerical instability and other reasons.

      DQN in DI-engine enables Double DQN by default, users can modify ``target_update_freq`` to control the update speed of target network, and code implementation is shown in ``TargetNetworkWrapper`` in `Double DQN code <https://github.com/opendilab/DI-engine/blob/main/ding/model/wrapper/model_wrappers.py>`_ .

    - Dueling head

      In `Dueling Network Architectures for Deep Reinforcement Learning <https://arxiv.org/abs/1511.06581>`_, dueling head architecture is utilized to implement the decomposition of state-value and advantage for taking each action, and use these two parts to construct the final q_value, which is better for evaluating the value of some states that show fewer connections with action selection.

      The specific architecture is shown in the following graph:

      .. image:: images/DuelingDQN.png
           :align: center
           :height: 300

      In DI-engine, users can enable Dueling head by modifying the ``dueling`` field in the model part of the configuration file, and the detailed code class ``DuelingHead`` is located in `Dueling Head <https://github.com/opendilab/DI-engine/blob/main/ding/model/common/head.py>`_.

    - RNN (DRQN, R2D2)

      For the combination of DQN and RNN, please refer to `R2D2 <./r2d2.html>`_ in this series doc.

Implementations
----------------
The default config of DQNPolicy is defined as follows:

.. autoclass:: ding.policy.dqn.DQNPolicy
   :noindex:

The network interface DQN used is defined as follows:

.. autoclass:: ding.model.template.q_learning.DQN
   :members: forward
   :noindex:


Benchmark
-----------


+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| environment         |best mean reward | evaluation results                                  | config link              | comparison           |
+=====================+=================+=====================================================+==========================+======================+
|                     |                 |                                                     |`config_link_p <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Tianshou(20)        |
|                     |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|Pong                 |  20             |.. image:: images/benchmark/pong_dqn.png             |atari/config/serial/      |                      |
|                     |                 |                                                     |pong/pong_dqn_config      |  Sb3(20)             |
|(PongNoFrameskip-v4) |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_q <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Tianshou(7307)      |
|Qbert                |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |  17966          |.. image:: images/benchmark/qbert_dqn.png            |atari/config/serial/      |  Rllib(7968)         |
|(QbertNoFrameskip-v4)|                 |                                                     |qbert/qbert_dqn_config    |                      |
|                     |                 |                                                     |.py>`_                    |  Sb3(9496)           |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_s <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Tianshou(812)       |
|SpaceInvaders        |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |  2403           |.. image:: images/benchmark/spaceinvaders_dqn.png    |atari/config/serial/      |  Rllib(1001)         |
|(SpaceInvadersNoFrame|                 |                                                     |spaceinvaders/space       |                      |
|skip-v4)             |                 |                                                     |invaders_dqn_config.py>`_ |  Sb3(622)            |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+


P.S.：

1. The above results are obtained by running the same configuration on five different random seeds (0, 1, 2, 3, 4)
2. For the discrete action space algorithm like DQN, the Atari environment set is generally used for testing (including sub-environments Pong), and Atari environment is generally evaluated by the highest mean reward training 10M ``env_step``. For more details about Atari, please refer to `Atari Env Tutorial <../env_tutorial/atari.html>`_ .


Reference
----------

- Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller: “Playing Atari with Deep Reinforcement Learning”, 2013; arXiv:1312.5602.

- Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016, June). Dueling network architectures for deep reinforcement learning. In International conference on machine learning (pp. 1995-2003). PMLR.

- Van Hasselt, H., Guez, A., & Silver, D. (2016, March). Deep reinforcement learning with double q-learning. In Proceedings of the AAAI conference on artificial intelligence (Vol. 30, No. 1).

- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). Prioritized experience replay. arXiv preprint arXiv:1511.05952.

