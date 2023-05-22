C51
======

概述
--------

C51 最初是在
`A Distributional Perspective on Reinforcement Learning <https://arxiv.org/abs/1707.06887>`__ 中提出的，与以往的研究不同，C51 评估了 q 值的完整分布，而不仅仅是期望值。作者设计了一个分布式 Bellman 算子，它保留了值分布中的多峰性，被认为能够实现更稳定的学习，并减轻从非稳态策略学习的负面影响。


核心要点
-----------

1. C51 是一种 **无模型（model-free）** 和 **基于值（value-based）** 的强化学习算法。

2. C51 仅 **支持离散动作空间** 。

3. C51 是一种 **异策略（off-policy）** 算法。

4. 通常, C51 使用 **eps-greedy** 或 **多项式采样** 进行探索。

5. C51 可以与循环神经网络 (RNN) 结合使用。

伪代码
------------
.. image:: images/C51.png
   :align: center
   :scale: 30%


.. note::
    C51 使用离散分布来建模值分布，其支持集合为N个原子: :math:`z_i = V_\min + i * delta, i = 0,1,...,N-1` 和　:math:`delta = (V_\max - V_\min) / N` 。每个原子　:math:`z_i` 都有一个参数化的概率 :math:`p_i` 。C51 的贝尔曼更新将 :math:`r + \gamma * z_j^{\left(t+1\right)}` 的分布投影到分布 :math:`z_i^t` 上。

关键方程或关键框图
---------------------------

C51 的贝尔曼方程的目标是通过将返回分布 :math:`r + \gamma * z_j` 投影到当前分布 :math:`z_i` 上来得到的. 给定一个采样出来的状态转移 :math:`(x, a, r, x')`，我们为每个原子　:math:`z_j` 计算贝尔曼更新 :math:`Tˆz_j := r + \gamma z_j` ，然后将其概率 :math:`p_{j}(x', \pi(x'))` 分配给其相邻的原子 :math:`p_{i}(x, \pi(x))`:

.. math::

   \left(\Phi \hat{T} Z_{\theta}(x, a)\right)_{i}=\sum_{j=0}^{N-1}\left[1-\frac{\left|\left[\hat{\mathcal{T}} z_{j}\right]_{V_{\mathrm{MIN}}}^{V_{\mathrm{MAX}}}-z_{i}\right|}{\Delta z}\right]_{0}^{1} p_{j}\left(x^{\prime}, \pi\left(x^{\prime}\right)\right)


扩展
-----------
- C51 可以和以下模块结合:
   - 优先经验回放 (Prioritized Experience Replay)
   - 多步时序差分 (TD) 损失
   - 双目标网络 (Double Target Network)
   - Dueling head
   - 循环神经网络 (RNN)

实现
-----------------

.. tip::
      我们的 C51 基准结果使用了与 DQN 相同的超参数，(除 `n_atom` 这个 C51 特有的参数以外)，这也是配置 C51 的通用方法。

C51 的默认配置如下:


.. autoclass:: ding.policy.c51.C51Policy
   :noindex:

C51 使用的网络接口定义如下:

.. autoclass:: ding.model.template.q_learning.C51DQN
   :members: forward
   :noindex:


基准测试
---------------------

.. list-table:: Benchmark and comparison of c51 algorithm
   :widths: 25 15 30 15 15
   :header-rows: 1

   * - environment
     - best mean reward
     - evaluation results
     - config link
     - comparison
   * - | Pong
       | (PongNoFrameskip-v4)
     - 20.6
     - .. image:: images/benchmark/c51_pong.png
     - `config_link_p <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/pong/pong_c51_config.py>`_
     - | Tianshou(20)
   * - | Qbert
       | (QbertNoFrameskip-v4)
     - 20006
     - .. image:: images/benchmark/c51_qbert.png
     - `config_link_q <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/qbert/qbert_c51_config.py>`_
     - | Tianshou(16245)
   * - | SpaceInvaders
       | (SpaceInvadersNoFrame skip-v4)
     - 2766
     - .. image:: images/benchmark/c51_spaceinvaders.png
     - `config_link_s <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/spaceinvaders/spaceinvaders_c51_config.py>`_
     - | Tianshou(988.5)

P.S.：

1. 上述结果是在五个不同的随机种子（0、1、2、3、4）上运行相同配置的实验得出的。
2. 对于像 DQN 这样的离散动作空间算法，通常使用 Atari 环境集进行测试（包括子环境 Pong ），而 Atari 环境通常通过训练10M个环境步骤的最高平均奖励来评估。关于 Atari 的更多细节，请参考 `Atari Env Tutorial <../env_tutorial/atari.html>`_  。


其他开源实现
-----------------------------
 - `Tianshou <https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/c51.py>`_

