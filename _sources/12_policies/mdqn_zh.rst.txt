MDQN
^^^^^^^

概述
---------
MDQN 是在 `Munchausen Reinforcement Learning <https://arxiv.org/abs/2007.14430>`_ 中提出的。 作者将这种通用方法称为 “Munchausen Reinforcement Learning”
(M-RL)， 以纪念 Raspe 的《吹牛大王历险记》中的一段著名描写， 即 Baron 通过拉自己的头发从沼泽中脱身的情节。
从实际使用的角度来看， MDQN 和 DQN 之间的关键区别是 Soft-DQN (传统 DQN 算法的扩展)的即时奖励中添加了一个缩放的 log-policy 。

核心要点
-------------
1。 MDQN 是一种 **无模型 (model-free)** 且 **基于值函数 (value-based)** 的强化学习算法。

2。 MDQN 只支持 **离散 (discrete)**  动作空间。

3。 MDQN 是一个 **异策略 (off-policy)** 算法。

4。 MDQN 使用 **epsilon贪心 (eps-greedy)** 来做探索 (exploration)。

5。 MDQN 增加了 **动作间隔 (action gap)** ， 并具有隐式的 **KL正则化 (KL regularization)** 。


关键方程或关键框图
---------------------------
MDQN 中使用的目标 Q 值 (target Q value) 是:

.. math::

   \hat{q}_{\mathrm{m} \text {-dqn }}\left(r_t, s_{t+1}\right)=r_t+\alpha \tau \ln \pi_{\bar{\theta}}\left(a_t \mid s_t\right)+\gamma \sum_{a^{\prime} \in A} \pi_{\bar{\theta}}\left(a^{\prime} \mid s_{t+1}\right)\left(q_{\bar{\theta}}\left(s_{t+1}, a^{\prime}\right)-\tau \ln \pi_{\bar{\theta}}\left(a^{\prime} \mid s_{t+1}\right)\right)
   

我们使用以下公式计算 log-policy 的值：  :math:`\alpha \tau \ln \pi_{\bar{\theta}}\left(a_t \mid s_t\right)` 

.. math::

   \tau \ln \pi_{k}=q_k-v_k-\tau \ln \left\langle 1, \exp \frac{q_k-v_k}{\tau}\right\rangle

其中  :math:`q_k`  在我们的代码中表示为  `target_q_current` 。 对于最大熵部分  :math:`\tau \ln \pi_{\bar{\theta}}\left(a^{\prime} \mid s_{t+1}\right)` 我们使用相同的公式进行计算，其中  :math:`q_{k+1}` 在我们的代码中表示为 `target_q` 。

我们将 :math:`\tau \ln \pi(a \mid s)` 替换为 :math:`[\tau \ln \pi(a \mid s)]_{l_0}^0`` 因为对数策略项 (log-policy term) 是无界的， 如果策略变得过于接近确定性策略 (deterministic policy) ，可能会导致数值性问题 (numerical issues) 。 

同时还将 :math:`\pi_{\bar{\theta}}\left(a^{\prime} \mid s_{t+1}\right)` 替换为 :math:`softmax(q-v)` ，因为这是在官方实现中使用的方法，但他们并未在论文中提及。

我们使用上述改动后的配置在 asterix 进行测试，得到了与原论文相同的结果， 即MDQN可以增加动作间隙 (action gap) 。

.. image:: images/action_gap.png
   :align: center

伪代码
---------------
.. image:: images/mdqn.png
   :align: center

扩展
---------------
- TBD


实现
----------------
MDQNPolicy 的默认配置如下：

.. autoclass:: ding.policy.mdqn.MDQNPolicy
   :noindex:


MDQN 使用的 TD error 接口定义如下：

.. autofunction:: ding.rl_utils.td.m_q_1step_td_error
   :noindex:


实验 Benchmark
------------------

.. list-table:: Benchmark and comparison of mdqn algorithm
   :widths: 25 15 30 15 15
   :header-rows: 1

   * - environment
     - best mean reward
     - evaluation results
     - config link
     - comparison
   * - | Asterix 
       | (Asterix-v0)
     - 8963
     - .. image:: images/benchmark/mdqn_asterix.png 
     - `config_link_asterix <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/asterix/asterix_mdqn_config.py>`_
     - | sdqn(3513) paper(1718) dqn(3444)
   * - | SpaceInvaders
       | (SpaceInvaders-v0)
     - 2211
     - .. image:: images/benchmark/mdqn_spaceinvaders.png
     - `config_link_spaceinvaders <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/spaceinvaders/spaceinvaders_mdqn_config.py>`_
     - | sdqn(1804) paper(2045) dqn(1228)
   * - | Enduro
       | (Enduro-v4)
     - 1003
     - .. image:: images/benchmark/mdqn_enduro.png
     - `config_link_enduro <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/enduro/enduro_mdqn_config.py>`_
     - | sdqn(986.1) paper(1171) dqn(986.4)



我们的配置和论文中的配置的主要区别如下：

-  我们收集了100个样本，进行了十次训练。而在原论文中，收集了4个样本，进行了一次训练。
-  我们每500个迭代更新一次目标网络 (target network) ，而原论文每2000个迭代更新一次目标网络。
-  我们用于探索的epsilon从1逐渐下降到0.05，而原论文的epsilon是从0.01到0.001。

P.S.:

-  以上结果是在 **seed 0** 上运行同样配置得到的。
-  对于像DQN这样的离散动作空间算法， 一般采用Atari环境集来进行测试， Atari 环境一般通过10M ``env_step`` 的最高均值奖励（highest mean reward）训练来评估。关于Atari环境的更多细节请参考：  `Atari 环境教程 <../env_tutorial/atari.html>`_ 

参考文献
----------

- Vieillard, Nino, Olivier Pietquin, and Matthieu Geist. "Munchausen reinforcement learning." Advances in Neural Information Processing Systems 33 (2020): 4235-4246.

