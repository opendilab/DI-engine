MBPO
^^^^^^^

概述
---------
Model-based policy optimization (MBPO) 首次在论文 `When to Trust Your Model: Model-Based Policy Optimization <https://arxiv.org/abs/1906.08253>`_ 中被提出。
MBPO 利用模型生成的短轨迹，并保证每一步的单调提升。
具体来说，MBPO 通过训练模型集合来拟合真实环境的 transition ，并利用它生成从真实环境状态开始的短轨迹来进行策略提升。
对于 RL 策略的选择，MBPO 使用 SAC 作为其 RL 的部分。

这个 repo `awesome-model-based-RL <https://github.com/opendilab/awesome-model-based-RL>`_ 提供了更多 model-based rl 的论文。


核心要点
-------------
1. MBPO 是一种 **基于模型（model-based）的** 强化学习算法。

2. MBPO 用 SAC 作为 RL 策略。

3. MBPO 仅支持 **连续动作空间** 。

4. MBPO 使用了 **model-ensemble**。


关键方程或关键框图
---------------------------

预测模型（Predictive Model）
::::::::::::::::::::::::::::

MBPO 利用高斯神经网络集合（ensemble of gaussian neural network），集合中的每个成员都是： 

.. math::

  p_\theta(\boldsymbol{s}_{t+1}|\boldsymbol{s}_t,\boldsymbol{a}_t) = N(\mu_\theta(\boldsymbol{s}_t,\boldsymbol{a}_t), \Sigma_\theta(\boldsymbol{s}_t,\boldsymbol{a}_t))

模型训练中使用的最大似然损失为：

.. math::

  L(\theta)=\mathbb{E}\left[log(p_\theta(\boldsymbol{s}_{t+1}|\boldsymbol{s}_t,\boldsymbol{a}_t))\right]


策略优化（Policy Optimization）
:::::::::::::::::::::::::::::::

策略评估步骤（Policy evaluation step）：

.. math::
  Q^\pi(\boldsymbol{s}_t,\boldsymbol{a}_t) = \mathbb{E}_\pi[{\sum}_{t=0}^{\infty}\gamma^t r(\boldsymbol{s}_t,\boldsymbol{a}_t)]

策略提升步骤（Policy improvement step）：

.. math::
  \min J_\pi(\phi, D) = \mathbb{E}_{s_t \sim D}[D_{KL}(\pi \| exp\{Q^\pi - V^\pi\})]

注意：这个更新要保证
:math:`Q^{\pi_{new}}(\boldsymbol{s}_t,\boldsymbol{a}_t) \geq Q^{\pi_{old}}(\boldsymbol{s}_t,\boldsymbol{a}_t)`，
可以查看原论文中 Appendix B.2 部分 Lemma2 的相关证明 `paper <https://arxiv.org/abs/1801.01290>`_。



伪代码
---------------
.. image:: images/MBPO.png
  :align: center
  :scale: 55%

.. note::
  MBPO 的首次实现只给出了应用于 SAC 的超参数，并不适用于 DDPG 和 TD3 。


实现
----------------
默认配置定义如下:

.. autoclass:: ding.policy.mbpolicy.mbsac.MBSACPolicy
   :noindex:



基准
-----------


.. list-table:: Benchmark of MBPO algorithm
   :widths: 25 30 15
   :header-rows: 1

   * - environment
     - evaluation results
     - config link
   * - Hopper
     - .. image:: images/benchmark/sac_mbpo_hopper.png
     - `config_link_p <https://github.com/opendilab/DI-engine/blob/main/dizoo/mujoco/config/hopper_sac_mbpo_default_config.py>`_
   * - Halfcheetah
     - .. image:: images/benchmark/sac_mbpo_halfcheetah.png
     - `config_link_q <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/halfcheetah_sac_mbpo_default_config.py>`_
   * - Walker2d
     - .. image:: images/benchmark/sac_mbpo_walker2d.png
     - `config_link_s <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/walker2d_sac_mbpo_default_config.py>`_


P.S.：

1. 上述结果是通过在三个不同的随机种子(0,1,2)上运行相同的配置获得的。


其他公开的实现
-------------------------------
- `mbrl-lib <https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/algorithms/mbpo.py>`_


参考文献
----------

- Michael Janner, Justin Fu, Marvin Zhang, Sergey Levine: “When to Trust Your Model: Model-Based Policy Optimization”, 2019; arXiv:1906.08253.
