QRDQN
^^^^^^^

概述
---------
QR (Quantile Regression, 分位数回归) DQN 在 `Distributional Reinforcement Learning with Quantile Regression <https://arxiv.org/pdf/1710.10044>`_  中被提出，它继承了学习 q 值分布的思想。与使用离散原子来近似分布密度函数不同， QRDQN 直接回归 q 值的一组离散分位数。


核心要点
-----------
1. QRDQN 是一种 **无模型（model-free）** 和 **基于值（value-based）** 的强化学习算法。

2. QRDQN 仅支持 **离散动作空间** 。

3. QRDQN 是一种 **异策略（off-policy）** 算法。

4. 通常情况下， QRDQN 使用 **eps-greedy** 或 **多项式采样** 进行探索。

5. QRDQN 可以与循环神经网络 (RNN) 结合使用。


关键方程或关键框图
----------------------------
C51 (Categorical 51) 使用N个固定位置来近似其概率分布，并调整它们的概率，而 QRDQN 将固定的均匀概率分配给N个可调整的位置。基于这一点， QRDQN 使用分位数回归来随机调整分布的位置，以使其与目标分布的 Wasserstein 距离最小化。

分位数回归损失是一种非对称凸损失函数，用于量化回归问题。对于给定的分位数 :math:`\tau \in [0, 1]` ，该损失函数以权重 :math:`\tau` 惩罚过估计误差，以权重 :math:`1−\tau` 惩罚欠估计误差. 
对于一个分布 :math:`Z` 和给定的分位数 :math:`\tau`，分位数函数 :math:`F_Z^{−1}(\tau)` 的值可以被描述为分位数回归损失的最小化器：

.. math::

   \begin{array}{r}
   \mathcal{L}_{\mathrm{QR}}^{\tau}(\theta):=\mathbb{E}_{\hat{z} \sim Z}\left[\rho_{\tau}(\hat{Z}-\theta)\right], \text { where } \\
   \rho_{\tau}(u)=u\left(\tau-\delta_{\{u<0\}}\right), \forall u \in \mathbb{R}
   \end{array}

上述提到的损失在零点处不平滑，这可能会限制在使用非线性函数逼近时的性能。因此，在 QRDQN 的 Bellman 更新过程中应用了一种修改后的分位数 Huber 损失， 称为 ``quantile huber loss`` 损失(即伪代码中的方程式10)。

.. math::

   \rho^{\kappa}_{\tau}(u)=L_{\kappa}(u)\lvert \tau-\delta_{\{u<0\}} \rvert

在这里 :math:`L_{\kappa}` 是 Huber 损失.

.. note::

   与 DQN 相比， QRDQN 具有以下区别:

     1. 神经网络架构: QRDQN 的输出层大小为M x N，其中M是离散动作空间的大小，N是一个超参数，表示分位数目标的数量。
     2. 使用分位数 Huber 损失替代 DQN 损失函数。
     3. 在原始的 QRDQN 论文中，将 RMSProp 优化器替换为 Adam 优化器。而在 DI-engine 中，我们始终使用 Adam 优化器。

伪代码
-------------
.. image:: images/QRDQN.png
   :align: center
   :scale: 25%

扩展
-----------
- QRDQN可以与以下技术相结合使用:

  - 优先经验回放 (Prioritized Experience Replay)
  - 多步时序差分 (TD)损失
  - 双目标网络 (Double Target Network)
  - 循环神经网络 (RNN)

实现
----------------

.. tip::
      在我们的基准结果中， QRDQN 使用与 DQN 相同的超参数，除了 QRDQN 的专属超参数——"分位数的数量" ，该超参数经验性地设置为32。

QRDQN 的默认配置可以如下定义:

.. autoclass:: ding.policy.qrdqn.QRDQNPolicy
   :noindex:

QRDQN 使用的网络接口可以如下定义:

.. autoclass:: ding.model.template.q_learning.QRDQN
   :members: forward
   :noindex:

QRDQN 的贝尔曼更新在ding/rl_utils/td.py模块的qrdqn_nstep_td_error函数中实现。

基准
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

P.S.:

1. 上述结果是通过在五个不同的随机种子 (0, 1, 2, 3, 4)上运行相同的配置获得的。
2. 对于像 QRDQN 这样的离散动作空间算法，通常使用 Atari 环境集进行测试(包括子环境 Pong ) ，而 Atari 环境通常通过训练10M个环境步骤的最高平均奖励来评估。有关 Atari 的更多详细信息, 请参阅 `Atari Env Tutorial <../env_tutorial/atari.html>`_ .

参考文献
------------

(QRDQN) Will Dabney, Mark Rowland, Marc G. Bellemare, Rémi Munos: “Distributional Reinforcement Learning with Quantile Regression”, 2017; arXiv:1710.10044. https://arxiv.org/pdf/1710.10044


其他开源实现
-------------------------------

- `Tianshou <https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/qrdqn.py>`_
