Guided Cost Learning
^^^^^^^^^^^^^^^^^^^^^^^

综述
---------
Guided Cost Learning(GCL)是一种逆强化学习算法，在 `Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization <https://arxiv.org/abs/1603.00448>`_ 中被提出。
Inverse Reinforcement Learning的一个基本思路就是，在给定的专家数据下，学习一个奖励函数，使得这个专家策略在这个奖励函数下是最优的，但这个奖励函数却并不是唯一的。在 `Maximum Entropy Inverse Reinforcement Learning <https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf>`_ 和 `Maximum Entropy Deep Inverse Reinforcement Learning <https://arxiv.org/abs/1507.04888>`_ 中提出了利用最大熵原理的方法来求奖励函数。GCL算法实在这些算法的基础上，利用最大熵原理，使用神经网络来表征一个非线性的奖励函数。GCL算法在训练过程中可以同时学习一个奖励函数reward function和策略policy。 GCL算法主要应用于控制领域，如机械臂控制等场合。

快速了解
-------------
1. GCL 是可以同时学习 **reward function** （奖励函数） 和 **policy** （策略） 的逆强化学习算法。

2. GCL 支持 **离散** 和 **连续** 动作空间。

3. GCL 算法可与 **PPO** 或者 **SAC** 等 **policy-based** 或 **Actor-Critic** 算法结合，同时学习reward function和与GCL相结合的policy。

4. GCL 学到的reward function的输入为 **state** 和 **action**，输出为估计的 **reward** 值。


重要公示/重要图示
---------------------------
GCL算法基于最大熵原理的一个基本公式：

.. math::

   p(\tau )=\frac{1}{Z} exp(-c_\theta(\tau))
   Z=\int exp(-c_\theta (\tau))d\tau 

其中 :math:`\tau` 指轨迹样本， :math:`p(\tau )`表示轨迹的概率， :math:`c_\theta(\tau)` 表示奖励函数算出的cost值，在这个模型下，次优的轨迹以指数递减的概率发生。

可以推出IRL的log-likelihood目标：

.. math::

   \mathcal{L}_{\mathrm{IOC}}(\theta)=\frac{1}{N} \sum_{\tau_{i} \in \mathcal{D}_{\text {demo }}} c_{\theta}\left(\tau_{i}\right)+\log Z\approx\frac{1}{N} \sum_{\tau_{i} \in \mathcal{D}_{\text {demo }}} c_{\theta}\left(\tau_{i}\right)+\log\frac{1}{M}\sum_{\tau_{i} \in \mathcal{D}_{\text {samp }}} \frac{exp(-c_{\theta}(\tau_j)) }{q(\tau_j)}

采用重要性采样的方法，可以记 :math: `w_j = \frac{exp(-c_{\theta}(\tau_j)) }{q(\tau_j)}`， `Z=\sum_{j}w_j` 可得：

.. math::

   \frac{d \mathcal{L}_{\mathrm{IOC}}}{d \theta}=\frac{1}{N} \sum_{\tau_{i} \in \mathcal{D}_{\text {demo }}} \frac{d c_{\theta}}{d \theta}\left(\tau_{i}\right)-\frac{1}{Z} \sum_{\tau_{j} \in \mathcal{D}_{\text {samp }}} w_{j} \frac{d c_{\theta}}{d \theta}\left(\tau_{j}\right)

使用该Loss函数来训练GCL算法的reward模型。

伪代码
---------------
.. image:: images/GCL_1.png
   :align: center
   :scale: 55%

.. image:: images/GCL_2.png
   :align: center
   :scale: 55%


扩展
-----------
GCL 可以和以下方法相结合：

    - PPO `Proximal Policy Optimization <https://arxiv.org/pdf/1707.06347.pdf>`_

    - SAC `Soft Actor-Critic <https://arxiv.org/pdf/1801.01290>`_

    使用PPO或SAC算法中Actor网络算出的概率值作为训练reward function时的轨迹概率 :math:`q(\tau )`, 使用GCL reward function算出的reward值作为训练PPO或SAC算法时的reward值。


实现
----------------
GCl 的默认 config 如下所示：

.. autoclass:: ding.dizoo.box2d.lunarlander.config.lunarlander_gcl_config.py
   :noindex:

其中使用的奖励模型接口如下所示：

.. autoclass:: ding.reward_model.guided_cost_reward_model.GuidedCostRewardModel
   :members: train, estimate
   :noindex:

实验 Benchmark
------------------


+---------------------+-----------------+-----------------------------------------------------+--------------------------+
| environment         |best mean reward | evaluation results                                  | config link              | 
+=====================+=================+=====================================================+==========================+
|                     |                 |                                                     |`config link <https://    |
|                     |                 |                                                     |github.com/opendilab/     |
|                     |                 |                                                     |DI-engine/tree/main/dizoo/|
|Lunarlander          |  2M env_step,   |.. image:: images/benchmark/lunarlander_gcl.png      |box2d/lunarlander/config/ |
|                     |  reward 200     |                                                     |lunarlander_gcl_config    |
|                     |                 |                                                     |.py>`_                    |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+
|                     |                 |                                                     |`config link <https://    |
|                     |                 |                                                     |github.com/opendilab/     |
|Hopper               |                 |                                                     |DI-engine/tree/main/dizoo/|
|                     |  3M  env_step,  |.. image:: images/benchmark/Hopper_gcl.png           |mujoco/config/            |
|                     |  reward 2950    |                                                     |.py>`_                    |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+



注：

1. 以上结果对比了生成专家模型的PPO算法和使用专家模型的GCL算法，对比了best mean reward和达到best mean reward所用的env_step



参考文献
----------

- Chelsea Finn, Sergey Levine, Pieter Abbeel: “Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization”, 2016; arXiv:1603.00448.


