DQN
^^^^^^^

综述
---------
DQN最初在论文 `Playing Atari with Deep Reinforcement Learning <https://arxiv.org/abs/1312.5602>`_ 中被提出。
传统的 Q-learning 维护一张 \ ``M*N`` \的Q值表（其中 M表示状态个数，N表示动作个数），通过贝尔曼方程（Bellman equation）来迭代更新 Q-value。这种算法在状态/动作空间变得很大的时候就会出现维度灾难的问题。而DQN与传统强化学习方法不同，它将 Q-learning 与深度神经网络相结合，使用深度神经网络来估计 Q 值，并通过计算时序差分（TD, Temporal-Difference） 损失，利用梯度下降算法进行更新，从而在高维空间的问题决策中（例如Atari游戏）达到了媲美甚至超过人类玩家的水平。

快速了解
-------------
1. DQN 是一个 **无模型（model-free)** 且 **基于值函数（value-based）** 的强化学习算法。

2. DQN 只支持 **离散（discrete）** 动作空间。

3. DQN 是一个 **异策略（off-policy）** 算法.

4. 通常，DQN 使用 **epsilon贪心（eps-greedy）** 或 **多项分布采样（multinomial sample）** 来做探索（exploration）。

5. DQN + RNN = DRQN

6. DI-engine 中实现的 DQN 支持 **多维度离散（multi-discrete）** 动作空间，即在一个step下执行多个离散动作。


重要公示/重要图示
---------------------------
DQN 中的 TD-loss 是：

.. math::

   L(w)=\mathbb{E}\left[(\underbrace{r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime}, w\right)}_{\text {Target }}-Q(s, a, w))^{2}\right]

伪代码
---------------
.. image:: images/DQN.png
   :align: center
   :scale: 55%

伪代码中的 equation（3） 指的是：

.. image:: images/eq3.png
   :align: center

.. note::
   DQN在发展过程中变更出了许多版本。与原始版本相比，现代的 DQN 在算法和实现方面都得到了显著改进。譬如，在算法部分，**TD-loss, PER, n-step, target network** and **dueling head** 等技巧被广泛使用，感兴趣的读者可参考论文 `Rainbow: Combining Improvements in Deep Reinforcement Learning <https://arxiv.org/abs/1710.02298>`_。在实现部分，探索所用的 epsilon 并非在整个训练流程保持不变，而是根据环境步数（envstep，意为策略与环境的交互次数），在训练期间，探索所用的 epsilon 从一个较高的初始值（比如，0.95）退火到一个较低值（比如，0.05）。

扩展
-----------
DQN 可以和以下方法相结合：

    - 优先级经验回放 （PER，`Prioritized Experience Replay <https://arxiv.org/abs/1511.05952>`_ ）

      Prioritized Experience Replay 用一种特殊定义的“优先级”来代替经验回放池中的均匀采样。该优先级可由各种指标定义，如绝对TD误差、观察的新颖性等。通过优先采样，DQN的收敛速度和性能可以得到很大的提高。

      优先级经验回放（PER）有很多种实现方式，其中一种较常用方式的伪代码如下图所示：

        .. image:: images/PERDQN.png
           :align: center
           :scale: 55%
      
      在DI-engine中，PER可以通过修改配置文件中的 ``priority`` 和 ``priority_IS_weight`` 两个字段来控制，具体的代码实现可以参考 `PER code <https://github.com/opendilab/DI-engine/blob/dev-treetensor/ding/worker/replay_buffer/advanced_buffer.py>`_ 。具体的示例讲解可以参考
      `PER example <../best_practice/priority.html>`_

    - 多步（Multi-step） TD-loss

      在 Single-step TD-loss 中，Q-learning 通过贝尔曼方程更新 :math:`Q(s,a)`:

        .. math::

          r(s,a)+\gamma \mathop{max}\limits_{a^*}Q(s',a^*)
      
      在 Multi-step TD-loss 中，贝尔曼方程是:

        .. math::
           \sum_{t=0}^{n-1}\gamma^t r(s_t,a_t) + \gamma^n \mathop{max}\limits_{a^*}Q(s_n,a^*)
        
        .. note::
          在DQN中使用 Multi-step TD-loss 有一个潜在的问题：采用 epsilon 贪心收集数据时， Q值的估计是有偏的。 因为t >= 1时，:math:`r(s_t,a_t)` 是在 epsilon-greedy 策略下采样的，而不是通过正在学习的策略本身来采样。但实践中发现 Multi-step TD-loss 与 epsilon-greedy 结合使用，一般都可以明显提升智能体的最终性能。

      在DI-engine中，Multi-step TD-loss 可以通过修改配置文件中的 ``nstep`` 字段来控制，详细的损失函数计算代码可以参考 `nstep code <https://github.com/opendilab/DI-engine/blob/dev-treetensor/ding/rl_utils/td.py>`_ 中的 ``q_nstep_td_error``

    - 目标网络（target network/Double DQN）

      Double DQN, 在 `Deep Reinforcement Learning with Double Q-learning <https://arxiv.org/abs/1509.06461>`_ 中被提出，是 DQN 的一种常见变种。此方法维护另一个 Q 网络，称为目标网络，该网络由当前网络按固定频率更新。

      Double DQN 中的 target Q 可以表示为：

        .. image:: images/doubleDQN.png
           :align: center
           :scale: 20%

      区别于传统DQN，Double DQN不会选择当前网络中离散动作空间中的最大Q值，而是首先查找  **当前网络**  中Q值最大的动作（对应上面公式中的  :math:`argmax_a Q(S_{t+1},a;\theta_t)`），然后根据该动作从 **目标网络**  获取Q值
      (对应上面公示中的  :math:`Q(S_{t+1},argmax_a Q(S_{t+1},a;\theta_t);\theta'_t)`）。
      
      Double DQN可以减少Q值过高估计的偏差，一定程度上解决由于这个过高估计所带来的的一系列问题。

        .. note::
            过高估计可能是由函数近似误差（近似Q值的神经网络）、环境噪声、数值不稳定等原因造成的。

      DI-engine实现的 DQN 中默认已经使用了Double DQN，可以通过修改 ``target_update_freq`` 字段来控制目标网络的更新速度，具体代码实现可以参考 `Double DQN code <https://github.com/opendilab/DI-engine/blob/main/ding/model/wrapper/model_wrappers.py>`_ 中的 ``TargetNetworkWrapper``

    - Dueling head (`Dueling Network Architectures for Deep Reinforcement Learning <https://arxiv.org/pdf/1511.06581>`_)

      Dueling head 结构通过对每个动作的状态-价值和优势的分解，并由上述两个部分构建最终的Q值，从而更好地评估一些与动作选择无关的状态的价值。下图展示了具体的分解结构（图片来自论文 Dueling Network Architectures for Deep Reinforcement Learning）：

        .. image:: images/DuelingDQN.png
           :align: center
           :height: 300

      在DI-engine中，Dueling head 可以通过修改配置文件中模型部分的 ``dueling`` 字段来控制，具体网络结构的实现可以参考 `Dueling Head <https://github.com/opendilab/DI-engine/blob/main/ding/model/common/head.py>`_ 中的 ``DuelingHead``

    - RNN (DRQN, R2D2) 
      
      DQN与RNN结合的方法，可以参考本系列文档中的 `R2D2部分 <./r2d2.html>`_

实现
----------------
DQNPolicy 的默认 config 如下所示：

.. autoclass:: ding.policy.dqn.DQNPolicy
   :noindex:

其中使用的神经网络接口如下所示：

.. autoclass:: ding.model.template.q_learning.DQN
   :members: forward
   :noindex:

实验 Benchmark
------------------


+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| environment         |best mean reward | evaluation results                                  | config link              | comparison           |
+=====================+=================+=====================================================+==========================+======================+
|                     |                 |                                                     |`config link <https://    |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Tianshou(20)        |
|                     |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|Pong                 |  20             |.. image:: images/benchmark/pong_dqn.png             |atari/config/serial/      |                      |
|                     |                 |                                                     |pong/pong_dqn_config      |  Sb3(20)             |
|(PongNoFrameskip-v4) |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config link <https://    |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Tianshou(7307)      |
|Qbert                |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |  17966          |.. image:: images/benchmark/qbert_dqn.png            |atari/config/serial/      |  Rllib(7968)         |
|(QbertNoFrameskip-v4)|                 |                                                     |qbert/qbert_dqn_config    |                      |
|                     |                 |                                                     |.py>`_                    |  Sb3(9496)           |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config link <https://    |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Tianshou(812)       |
|SpaceInvaders        |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |  2403           |.. image:: images/benchmark/spaceinvaders_dqn.png    |atari/config/serial/      |  Rllib(1001)         |
|(SpaceInvadersNoFrame|                 |                                                     |spaceinvaders/space       |                      |
|skip-v4)             |                 |                                                     |invaders_dqn_config.py>`_ |  Sb3(622)            |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+


注：

1. 以上结果是在5个不同的随机种子（即0，1，2，3，4）运行相同的配置得到
2. 对于DQN这样的离散动作空间算法，一般选择Atari环境集进行测试（其中包括子环境Pong等），而Atari环境，一般是通过训练10M个env_step下所得的最高平均奖励来进行评价，详细的环境信息可以查看 `Atari环境的介绍文档 <../env_tutorial/atari_zh.html>`_

参考文献
----------

- Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller: “Playing Atari with Deep Reinforcement Learning”, 2013; arXiv:1312.5602.

- Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016, June). Dueling network architectures for deep reinforcement learning. In International conference on machine learning (pp. 1995-2003). PMLR.

- Van Hasselt, H., Guez, A., & Silver, D. (2016, March). Deep reinforcement learning with double q-learning. In Proceedings of the AAAI conference on artificial intelligence (Vol. 30, No. 1).

- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). Prioritized experience replay. arXiv preprint arXiv:1511.05952.

