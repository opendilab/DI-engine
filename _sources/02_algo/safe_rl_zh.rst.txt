安全强化学习
====================

问题定义和研究动机
---------------------

**安全强化学习** （Safe Reinforcement Learning），是强化学习的一个细分发展方向。强化学习的核心目标是学习一个最大化预期回报的策略，但是在现实世界的决策问题，例如自动驾驶和机器人场景中，部署这种仅最大化预期回报的智能体通常会引发安全问题。

在强化学习训练过程中通常会出现以下安全问题 [[1]_]：

- 对环境的负面影响（Negative Side Effects）。

- 发掘奖励函数漏洞（Reward Hacking）。
  
- 信息有限时不采取错误动作（Scalable Oversight）。
  
- 安全地探索（Safe Exploration）。
  
- 对新环境或数据分布的安全性（Robustness to Distributional Shift）。

由于这些问题的存在，在实际部署时考虑安全条件时非常必要的。而在定义安全强化学习时有以下五个关键问题 [[2]_]：

- 安全策略（Safety Policy）。如何进行策略优化，寻找安全的策略？
  
- 安全复杂性（Safety Complexity）。 需要多少训练数据才能找到安全的策略？
  
- 安全应用（Safety Applications）。 安全强化学习应用的最新进展如何？
  
- 安全基准（Safety Benchmarks）。 我们可以使用哪些基准来公平全面地检查安全强化学习性能？
  
- 安全挑战（Safety Challenges）。 未来安全强化学习研究面临哪些挑战？
  
一个从统一性角度概述这五个安全强化学习问题的框架如下图所示 [[2]_]。

.. image:: images/safe_rl_2h3w.png
   :align: center
   :scale: 50 %

安全强化学习通常被建模为 **约束马尔科夫决策过程** （Constrained Markov Decision Process，CMDP），约束马尔科夫决策过程是马尔科夫决策过程（MDP）的扩展，由七元组 :math:`(S, A, P, r, c, b, \mu)`，即状态空间、动作空间、状态转移函数、奖励、代价、代价阈值、折扣因子组成。智能体采取动作后不仅会收到奖励 r 还会得到代价 c ，策略目标是在不超过代价阈值 b 的约束条件下最大化长期奖励：

\ :math:`\max_{\pi}\mathbb{E}_{\tau\sim\pi}\big[R(\tau)\big],\quad s.t.\quad\mathbb{E}_{\tau\sim\pi}\big[C(\tau)\big]\leq\kappa.`

.. image:: images/safe_gym.gif
   :align: center
   :scale: 50 %

上图是 OpenAI 发布的 `safety-gym <https://github.com/openai/safety-gym>`__ 环境，传统强化学习训练出的最优策略往往以任务为中心，不考虑对环境和自身的影响、不会考虑是否符合人类预期等等。小车（红色）会以最快速度移动到目标地点（绿色圆柱体），完全没有避开地面的陷阱区域（蓝色圆圈），移动路径上如果有障碍物（青色立方体）则会撞开或强行从边缘擦过。

研究方向
--------

安全强化学习的理论基础主要是对偶法和凸优化相关知识。主要的理论方法可以分为：

- 原问题对偶化（Primal Dual）方法，使用拉格朗日乘子法转换成解对偶问题。
- 原问题（Primal）方法，使用其他方式求解原问题。
  
而从训练方案上又可以分为：

- 无环境模型（Model-free）
- 基于环境模型（Model-based）

当前安全强化学习领域中算法的细致分类，可以参见下图（摘自 `omnisafe <https://github.com/PKU-Alignment/omnisafe>`__ ）：

.. image:: images/safe_rl_registry.png
   :align: center
   :scale: 50 %

原问题对偶化（Primal Dual） 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 safe RL 的原问题中，目标和约束都不是凸的，但是可以使用拉格朗日乘子法转换成解对偶问题，对偶问题是极小化凸问题，是可以求解的，这种方案有很多经典工作 [[3, 4, 5, 6, 7]_]。

拉格朗日函数：:math:`\mathcal{L}(\pi,\lambda)=V(\pi)+\Sigma\lambda_i(U_i(\pi)-c_i),\lambda\geq0`  

拉格朗日对偶函数：:math:`d(\lambda)=\max_{\pi\in\mathcal{P}(\mathcal{S})}\mathcal{L}(\pi,\lambda)`  

最小化对偶函数：:math:`D^*=\min_{\lambda\in\mathbb{R}_+}d(\lambda)` 就可以得到对偶问题最优解。


原问题（Primal） 
~~~~~~~~~~~~~~~~~~~

使用对偶化方案虽然很好地保证了问题的可解性，但是训练迭代的速度很慢，在优化策略函数的同时还要优化对偶函数，同时在选择拉格朗日乘子时并不轻松。因此有些方法不直接关注整个原问题的求解，而去利用 Natural Policy Gradiants 中的单步更新公式：  

.. image:: images/safe_rl_npg.png    
   :align: center    
   :scale: 50 %  
   
在每一步更新时求解一个比较简单的单步约束优化问题，保证每一次更新都不违法约束并提升表现，自然最终会得到一个符合约束的解。代表方法是 PCPO 等。


无环境模型（Model-free）
~~~~~~~~~~~~~~~~~~~~~~~~~~

约束策略优化（CPO）[3]是第一个解决CMDP问题的策略梯度方法。优化以下两式就可以实现保证回报单调增加的同时满足安全约束。

\ :math:`J\left(\pi'\right)-J(\pi)\geq\frac{1}{1-\gamma}\underset{\stackrel{s\sim d\pi}{a\sim\pi'}}{\operatorname*{E}}\left[A^{\pi}(s,a)-\frac{2\gamma\epsilon^{\pi'}}{1-\gamma}D_{TV}\left(\pi'\|\pi\right)[s]\right]`

\ :math:`J_{C_{i}}\left(\pi^{\prime}\right)-J_{C_{i}}\left(\pi\right)\leq\frac{1}{1-\gamma}\underset{\overset{s\sim d^{\pi}}{a\sim\pi^{\prime}}}{\operatorname*{E}}\left[A_{C_{i}}^{\pi}\left(s,a\right)+\frac{2\gamma\epsilon_{C_{i}}^{\pi^{\prime}}}{1-\gamma}D_{TV}\left(\pi^{\prime}\|\pi\right)\left[s\right]\right]`

这一方法几乎可以收敛到安全界限，并在某些任务上产生比原始对偶方法更好的性能。然而，CPO 的计算代价比拉格朗日乘子法更昂贵，因为它需要计算 Fisher 信息矩阵并使用二次泰勒展开来优化目标。

基于环境模型（Model-based）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

基于模型的深度强化学习 (DRL) 方法通常比无模型的 DRL 方法具有更好的学习效率，在安全强化学习领域同样是如此。但在现实情况下，构建准确的环境模型是具有挑战性的，许多模拟环境难以构建模型来辅助深度强化学习方法的部署。


未来展望
--------

当我们在实际应用中利用强化学习时，部署过程中会遇到许多挑战。安全强化学习是一个发展时间并不太久的方向，很多方面都有许多需要探索之处 [[8, 9, 10]_]。比如：

- 利用博弈论实现安全多智能体强化学习（safe-MARL）。可以在不同的游戏设置中考虑不同的游戏用于现实世界的应用。
- 基于信息论的安全强化学习。信息论可能有助于处理不确定性奖励信号和成本估计，并有效解决大规模多智能体环境的问题。
- 利用人脑理论和生物学理论。从生物学定律中汲取一些灵感来设计安全的强化学习算法。
- 人机交互。从与非专家用户的互动中学习，建模人类行为和现实交互，使机器人安全地继承人类偏好并更多地了解人类的潜在解决方案。


参考文献
--------

.. [1] Amodei D, Olah C, Steinhardt J, et al. Concrete problems in AI safety[J]. arXiv preprint arXiv:1606.06565, 2016.

.. [2] Gu S, Yang L, Du Y, et al. A review of safe reinforcement learning: Methods, theory and applications[J]. arXiv preprint arXiv:2205.10330, 2022.

.. [3] Achiam J, Held D, Tamar A, et al. Constrained policy optimization[C]//International conference on machine learning. PMLR, 2017: 22-31.

.. [4] Paternain, S., Calvo-Fullana, M., Chamon, L. F., & Ribeiro, A. (2019). Safe policies for reinforcement learning via primal-dual methods.arXiv preprint arXiv:1911.09101.

.. [5] Ding, D., Wei, X., Yang, Z., Wang, Z., & Jovanovic, M. (2021, March). Provably efficient safe exploration via primal-dual policy optimization. InInternational Conference on Artificial Intelligence and Statistics(pp. 3304-3312). PMLR.

.. [6] Ding, D., Zhang, K., Basar, T., & Jovanovic, M. R. (2020). Natural Policy Gradient Primal-Dual Method for Constrained Markov Decision Processes. InNeurIPS.

.. [7] Paternain, S., Chamon, L. F., Calvo-Fullana, M., & Ribeiro, A. (2019). Constrained reinforcement learning has zero duality gap.arXiv preprint arXiv:1910.13393.

.. [8] https://zhuanlan.zhihu.com/p/407168691

.. [9] https://zhuanlan.zhihu.com/p/347272765

.. [10] https://github.com/PKU-Alignment/omnisafe
