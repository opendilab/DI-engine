FQF
====

概述
---------
FQF 首次在论文 `Fully Parameterized Quantile Function for Distributional Reinforcement Learning <https://arxiv.org/pdf/1911.02140>`_ 中被提出。 FQF 和 IQN (Implicit Quantile Networks for Distributional Reinforcement Learning) 的主要区别在于，FQF 额外引入了 the fraction proposal network，这是一个被训练的参数函数，用于在[0, 1]范围内生成 tau, 而 IQN 是从一个 base distribution, 如 U([0, 1]) 对 tau 进行采样。

核心要点
-----------
1. FQF 是一种 **无模型（model-free）** and **基于值（value-based）** 的值分布强化学习算法。

2. FQF 仅支持 **离散动作空间** 。

3. FQF 是一种 **异策略（off-policy）** 算法。

4. 通常情况下， FQF 使用 **eps-greedy** 或 **多项式采样（multinomial sample）** 进行探索。

5. FQF 可以与循环神经网络 (RNN) 结合使用。

关键方程或关键框图
----------------------------
对于任何非递减的连续 quantile function :math:`F_{Z}^{-1}` , 定义 :math:`F_{Z}^{-1}` 与 :math:`F_{Z}^{-1, \tau}` 的 1-Wasserstein loss 为：

.. math::

    W_{1}(Z, \tau)=\sum_{i=0}^{N-1} \int_{\tau_{i}}^{\tau_{i+1}}\left|F_{Z}^{-1}(\omega)-F_{Z}^{-1}\left(\hat{\tau}_{i}\right)\right| d \omega

注意由于 :math:`W_{1}` 无法被计算出来, 我们不能直接对 the fraction proposal network 进行梯度下降。 取而代之地, 我们把 :math:`\frac{\partial W_{1}}{\partial \tau_{i}}` 
作为损失函数交给优化器去优化。

:math:`\frac{\partial W_{1}}{\partial \tau_{i}}` 由下式计算：

.. math::

    \frac{\partial W_{1}}{\partial \tau_{i}}=2 F_{Z}^{-1}\left(\tau_{i}\right)-F_{Z}^{-1}\left(\hat{\tau}_{i}\right)-F_{Z}^{-1}\left(\hat{\tau}_{i-1}\right), \forall i \in(0, N).

类似 implicit quantile networks, quantile tau 通过下式被编码进一个 embedding 向量：

.. math::

        \phi_{j}(\tau):=\operatorname{ReLU}\left(\sum_{i=0}^{n-1} \cos (\pi i \tau) w_{i j}+b_{j}\right)

然后将 the quantile embedding 与环境观测的 embedding 进行元素相乘，随后的全连接层将得到的乘积向量映射到相应的 quantile value 。

FQF 比 IQN 的优势如下图所示：左图 (a) 是 FQF 通过学习得到的 `\tau` ，右图 (b) 是 IQN 随机选择的 `\tau` ，阴影部分面积即为 1-Wasserstein loss，可以看出 FQF 得到 `\tau` 的方式要比 IQN 得到 `\tau` 的方式产生的 1-Wasserstein loss 要小。

.. image:: images/fqf_iqn_compare.png
   :align: center
   :scale: 100%

伪代码
-------------
.. image:: images/FQF.png
   :align: center
   :scale: 100%

扩展
-----------
FQF 可以与以下技术相结合使用:

  - 优先经验回放 (Prioritized Experience Replay)

    .. tip::
        是否优先级经验回放 (PER) 能够提升 FQF 的性能取决于任务和训练策略。

  - 多步时序差分 (TD) 损失
  - 双目标网络 (Double Target Network)
  - 循环神经网络 (RNN)

实现
------------------

.. tip::
      我们的 FQF 基准结果使用与DQN相同的超参数，除了 FQF 的独有超参数， ``the number of quantiles``， 它经验性地设置为32。直观地说，与随机 fractions 相比，trained quantile fractions 的优势在较小的 N 下更容易被观察到。在较大的 N 下，当 trained quantile fractions 和随机 fractions 都密集地分布在[0, 1]时， FQF 和 IQN 之间的差异变得可以忽略不计。

FQF 算法的默认配置如下所示：

.. autoclass:: ding.policy.fqf.FQFPolicy
   :noindex:

FQF 算法使用的网络接口定义如下：

.. autoclass:: ding.model.template.q_learning.FQF
   :members: forward
   :noindex:

FQF 算法中使用的贝尔曼更新（Bellman update）在 ``fqf_nstep_td_error`` 函数中定义，我们可以在 ``ding/rl_utils/td.py`` 文件中找到它。

基准测试
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
  1. 上述结果是通过在三个不同的随机种子 (0, 1, 2) 上运行相同的配置获得的。

参考文献
------------


(FQF) Derek Yang, Li Zhao, Zichuan Lin, Tao Qin, Jiang Bian, Tieyan Liu: “Fully Parameterized Quantile Function for Distributional Reinforcement Learning”, 2019; arXiv:1911.02140. https://arxiv.org/pdf/1911.02140


其他开源实现
---------------------------------

- `Tianshou <https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/fqf.py>`_
