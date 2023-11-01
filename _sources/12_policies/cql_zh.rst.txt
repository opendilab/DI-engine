CQL
^^^^^^^

综述
---------
离线强化学习（RL）是一个新出现的研究领域，旨在使用大量预先收集的数据集学习行为，而无需进一步与环境进行交互。它有可能在许多实际决策问题中取得巨大进展，
其中与环境交互昂贵（例如，在机器人技术、药物发现、对话生成、推荐系统中）或不安全/危险（例如，在医疗保健、自动驾驶或教育中）。
此外，在线收集的数据量远远低于离线数据集。这样的范式有望解决将强化学习算法从受限实验室环境带到现实世界的关键挑战。

然而，直接在离线设置中使用现有的基于价值的离线 RL 算法通常会导致性能不佳，这是由于从分布外动作（out-of-distribution actions）引导和过度拟合等问题。因此，许多约束技术被添加到基本在线 RL 算法中。 
保守 Q 学习（CQL），首次提出于 `Conservative Q-Learning for Offline Reinforcement Learning <https://arxiv.org/abs/2006.04779>`_,
是其中之一，它通过对标准基于价值的 RL 算法进行简单修改来学习保守 Q 函数，其期望值下限。

快速了解
-------------
1. CQL 是一种离线 RL 算法。

2. CQL 可以在许多标准在线 RL 算法之上用不到20行代码实现。

3. CQL 支持离散和连续动作空间。

重要公式/重要图示
---------------------------
CQL 可以在许多标准在线 RL 算法之上用不到20行代码实现，只需将 CQL 正则化项添加到 Q 函数更新中。

通常情况下，对于保守的离线策略评估，Q 函数通过迭代更新进行训练：

.. image:: images/cql_policy_evaluation.png
   :align: center
   :scale: 55%

仔细观察上面的方程，它由两部分组成-正则化项和通常的贝尔曼误差与权衡因子 :math:`\alpha` 。在正则化项内部，
第一项总是在从 :math:`\mu` 采样的（s,a）对上推动 Q 值下降，而第二项在从离线数据集抽取的（s,a）样本上推动Q值上升。

根据以下定理，当 :math:`\mu` = :math:`\pi` 时，上述方程下限了策略 :math:`\pi` 下的期望值。

对于合适的 :math:`\alpha` ，在采样误差和函数近似下，该界限成立。我们还注意到，随着更多数据变得可用并且\|D(s; a)\|增加，保证下界所需的 :math:`\alpha` 的理论值减小，
这表明在无限数据的极限情况下，可以通过使用极小的 :math:`\alpha` 值获得下界。

请注意，下面提供的分析假定 Q 函数中未使用函数近似，这意味着每次迭代都可以精确表示。该定理中的结果可以进一步推广到线性函数逼近器和非线性神经网络函数逼近器的情况，
其中后者基于 neural tangent kernel（NTK）框架。有关更多详细信息，请参阅原始论文附录 D.1 中的定理 D.1 和定理 D.2。

那么，我们应该如何利用这一点进行策略优化呢？我们可以在每个策略迭代 :math:`\hat{\pi}^{k}(a|s)` 之间交替执行完整的离线策略评估和一步策略改进。
然而，这可能会计算昂贵。另外，由于策略 :math:`\hat{\pi}^{k}(a|s)` 通常源自Q函数，我们可以选择 :math:`\mu(a|s)` 来近似最大化当前Q函数迭代的策略，从而产生一个在线算法。
因此，对于一个完整的离线RL算法，Q函数通常按如下方式更新：

.. image:: images/cql_general_3.png
   :align: center
   :scale: 55%

其中 :math:`CQL(R)` 由正则化器 :math:`R(\mu)` 的特定选择来表征。如果 :math:`R(\mu)` 被选择为与先验分布 :math:`\rho(a|s)` 的 KL 散度，
则我们得到 :math:`\mu(a|s)\approx \rho(a|s)exp(Q(s,a))` 。首先，如果 :math:`\rho(a|s)` = Unif(a)，则上面的第一项对应于任何状态 s 下 Q 值的软最大值，并产生以下变体，
称为CQL(H)：

.. image:: images/cql_equation_4.png
   :align: center
   :scale: 55%

其次，如果 :math:`\rho(a|s)` 被选择为前一个策略 :math:`\hat{\pi}^{k-1}` ，则上述方程（4）中的第一项被替换为来自所选 :math:`\hat{\pi}^{k-1}(a|s)` 的动作的 Q 值的指数加权平均值。

伪代码
---------------
伪代码显示在算法1中，与传统的 Actor-Critic 算法（例如SAC）和深度 Q 学习算法（例如DQN）的区别以红色显示。

.. image:: images/cql.png
   :align: center
   :scale: 55%

上述伪代码中的方程4如下：

.. image:: images/cql_equation_4.png
   :align: center
   :scale: 40%

请注意，在实现过程中，方程（4）中的第一项将在 `torch.logsumexp` 下计算，这会消耗大量运行时间。

实现
----------------
CQL 算法的默认设置如下：

.. autoclass:: ding.policy.cql.CQLPolicy
   :noindex:


.. autoclass:: ding.policy.cql.DiscreteCQLPolicy
   :noindex:

Benchmark
-----------

+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| environment         |best mean reward | evaluation results                                  | config link              | comparison           |
+=====================+=================+=====================================================+==========================+======================+
|                     |                 |                                                     |`config_link_ha <https:// |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|                     |                 |                                                     |DI-engine/blob/main/dizoo/|                      |
|HalfCheetah          |  57.6           |.. image:: images/benchmark/halfcheetah_cql.png      |d4rl/config/halfcheetah_  |   CQL Repo (75.6     |
|                     |  :math:`\pm`    |                                                     |cql_medium_expert         |   :math:`\pm` 25.7)  |
|(Medium Expert)      |  3.7            |                                                     |_config.py>`_             |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_w <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|Walker2d             |                 |                                                     |DI-engine/blob/main/dizoo/|                      |
|                     |  109.7          |.. image:: images/benchmark/walker2d_cql.png         |d4rl/config/walker2d_     |   CQL Repo (107.9    |
|(Medium Expert)      |  :math:`\pm`    |                                                     |cql_medium_expert         |   :math:`\pm` 1.6)   |
|                     |  0.8            |                                                     |_config.py>`_             |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_ho <https:// |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|Hopper               |                 |                                                     |DI-engine/blob/main/dizoo/|                      |
|                     |  85.4           |.. image:: images/benchmark/hopper_cql.png           |d4rl/config/hopper_sac_   |    CQL Repo (105.6   |
|(Medium Expert)      |  :math:`\pm`    |                                                     |cql_medium_expert         |    :math:`\pm` 12.9) |
|                     |  14.8           |                                                     |_config.py>`_             |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+

对于每个数据集，我们的实现结果如下：

+---------------------+-----------------+----------------+-----------------+-----------------+----------------------+
| environment         |random           |medium replay   |medium expert    |medium           |expert                |
+=====================+=================+================+=================+=================+======================+
|                     |                 |                |                 |                 |                      |
|HalfCheetah          |18.7 :math:`\pm` |47.1 :math:`\pm`|57.6 :math:`\pm` |49.7 :math:`\pm` |75.1 :math:`\pm`      |
|                     |1.2              |0.3             |3.7              |0.4              |18.4                  |
+---------------------+-----------------+----------------+-----------------+-----------------+----------------------+
|                     |                 |                |                 |                 |                      |
|Walker2d             |22.0 :math:`\pm` |82.6 :math:`\pm`|109.7 :math:`\pm`|82.4 :math:`\pm` |109.2 :math:`\pm`     |
|                     |0.0              |3.4             |0.8              |1.9              |0.3                   |
+---------------------+-----------------+----------------+-----------------+-----------------+----------------------+
|                     |                 |                |                 |                 |                      |
|Hopper               |3.1 :math:`\pm`  |98.3 :math:`\pm`|85.4 :math:`\pm` |79.6 :math:`\pm` |105.4  :math:`\pm`    |
|                     |2.6              |1.8             |14.8             |8.5              |7.2                   |
+---------------------+-----------------+----------------+-----------------+-----------------+----------------------+

P.S.：

1. 上述结果是通过在四个不同的随机种子（5、10、20、30）上运行相同的配置获得的。
2. 上述基准测试是针对HalfCheetah-v2、Hopper-v2、Walker2d-v2。
3. 上述比较结果是通过论文 `Pessimistic Bootstrapping for Uncertainty-Driven Offline Reinforcement Learning  <https://openreview.net/pdf?id=Y4cs1Z3HnqL>`_.
   获得的。完整表格如下所示。

   .. image:: images/cql_official.png
      :align: center
      :scale: 40%  


4. 上图给出了没有进行归一化（可以直接通过 env.get_normalized_score 函数得到）的结果。

引用
----------

- Kumar, Aviral, et al. "Conservative q-learning for offline reinforcement learning." arXiv preprint arXiv:2006.04779 (2020).
- Chenjia Bai, et al. "Pessimistic Bootstrapping for Uncertainty-Driven Offline Reinforcement Learning."


其他开源实现
----------------------------

- `CQL release repo`_


.. _`CQL release repo`: https://github.com/aviralkumar2907/CQL
