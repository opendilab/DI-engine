Basic Concepts
^^^^^^^^^^^^^^^

强化学习 (RL) 方法被用于解决智能体与环境的交互问题。简单的描述交互 (interaction) 的过程，智能体从环境中接收观察到的信息 (observation)，根据接收到的信息选择动作 (action)，环境会因为智能体采取的动作发生改变，并给予智能体动作的反馈奖励 (reward)。这个过程循环发生，智能体的目标是最大化积累自己接收到的奖励。强化学习的目标也就是让智能体学会一种策略 (policy) ，使奖励最大化。

为了更详细地介绍强化学习的理论细节与技术，我们会解释以下的基础概念：

- Markov Decision Processes 
- State and action spaces
- Policy
- Trajectory
- Return and reward

为了更好地理解各种强化学习的方法，我们会进一步解释以下的方法概念：

- RL optimization problem
- Value function
- Policy gradients
- Actor Critic

最后我们还解答了一些强化学习领域中常见的概念性问题，以供参考。

马尔可夫决策过程/MDP
------------------
**马尔可夫决策过程(Markov Decision Processes)**是强化学习在数学上的理想化形式，也是最常见的常见模型。

- 马尔可夫性：状态 :math:`s_t` 是马尔科夫的，当且仅当 :math:`P[s_{t+1}|s_t] = P[s_{t+1}|s_1, ..., s_t]` .
- 若随机变量序列中的每个状态都是马尔科夫的则称此随机过程为马尔科夫随机过程。
- 马尔科夫过程是一个二元组 :math:`(S, P)` ，且满足： :math:`S` 是有限状态集合， :math:`P` 是状态转移概率。马尔科夫过程中不存在动作和奖励。将动作（策略）和回报考虑在内的马尔科夫过程称为马尔科夫决策过程。
- 马尔科夫决策过程由元组 :math:`(S, A, P, R, \gamma)` 定义， :math:`S` 为有限的状态集， :math:`A` 为有限的动作集， :math:`P` 为状态转移概率， :math:`R` 为回报函数， :math:`\gamma` 为折扣因子，用来计算累积的奖励。跟马尔科夫过程不同的是，马尔科夫决策过程的状态转移概率为 :math:`P(s_{t+1}|s_t, a_t)` 。
- 强化学习的目标是给定一个马尔科夫决策过程，寻找最优策略。所谓策略 :math:`\pi(a|s)` 是指状态到动作的映射。

解决MDP问题的常用方法：

1. **动态规划 (DP)** 是一类优化方法，在给定一个MDP完备环境的情况下，可以计算最优的策略。但是对于强化学习问题，传统DP的作用有限，容易出现维度灾难问题。
   DP具有以下的特点：

   - 更新时基于当前已存在的估计：用后继各个状态的价值估计值来更新当前某个状态的价值估计值
   - 渐进性收敛
   - 优点：降低了方差并加快了学习
   - 缺点：存在依赖于函数逼近质量的偏差

2. **蒙特卡洛方法 (MC)** 用样本回报代替实际的期望回报，仅仅需要经验就可以求解最优策略。

   MC不需要环境模型，可以使用数据仿真和采样模型，并且可以只评估关注的某个状态，相比较DP在马尔可夫性不成立时损失较小。

3. **时序差分学习(TD)**, TD loss的基本形式: :math:`\delta_{t} = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)`
   TD与MC的对比: MC更新的目标是 :math:`G_t` 即时刻t的真实回报， 而TD(此时讨论单步TD即TD(0))更新的目标是 :math:`R_{t+1} + \gamma V(S_{t+1})` .


状态空间/State Spaces
--------------------
State :math:`s` 是对环境的global性描述，observation :math:`o`是对环境的部分性描述。一般的环境会使用实值向量、矩阵或高阶张量来表示状态和观察的结果。例如，Atari游戏中使用RGB图片来表示游戏环境的信息，MuJoCo控制任务中使用向量来表示智能体的状态。

当智能体能够接收到环境全部的状态信息 :math:`s` 时，我们称智能体的学习过程为fully observable，当智能体只能接收部分环境信息 :math:`o`时，我们称这个过程为partial observable的，即部分可观察的马尔可夫决策过程 (partially observable Markov decision processes，POMDP)，组成部分为  :math:`(O, A, P, R, \gamma)` .

动作空间/Action Spaces
---------------------
不同的环境所允许的动作空间不同。一般将环境中所有有效动作的集合称之为动作空间 (Action Space)。其中，动作空间又分为离散 (discrete) 动作空间与连续 (continuous) 动作空间。如在Atari游戏与SMAC星际游戏中为离散有限的动作空间，而MuJoCo等一些机器人连续控制任务中为连续动作空间，动作空间一般为实值向量区间。

策略/Policy
-----------


轨迹/Trajectory
---------------


奖励/Return and reward
---------------------


优化/RL optimization problem
------------------------


价值函数/Value functions
-----------------------



**Bellman Equations**


策略梯度/Policy Gradients
------------------------



演员-评论家/Actor Critic
-----------------------

**Actor**

**Critic**




Q&A
----
Q1: 什么是model base和model free，两者区别是什么？MC、TD、DP三者中哪些是model free，哪些是model based？
 - Answer：
   model base算法指该算法会学习环境的转移过程并对环境进行建模，而model free算法则不需要对环境进行建模。
   蒙特卡洛和TD算法隶属于model-free，因为这两个算法不需要算法建模具体环境。
   而动态规划属于model-based，因为使用动态规划需要完备的环境模型。

Q2: 什么是value-based， policy-based和collector-critic？ 分别有哪些算法是value-based，policy-based和actor-critic的？他们分别有什么advantage？有哪些drawback？
 - Answer：
   所谓value-based就是在学习如何critic(评判一个输入状态的价值)，policy-based对应的是学习如何去做actor(判断在一个输入状态应该采取什么行动)，而actor-critic就是一边去学习如何判断critic，一边去训练做actor的网络。
   具体关系用下图就能很好解释：
      
.. image:: images/actor-critic.jpg
   :scale: 30 %

Q3: 什么是on-policy和off-policy？
 - Answer：on-policy是使用当前的策略进行训练，用于生成采样数据序列的策略和用于实际决策的待评估和改进策略是相同的。 
   off-policy则是可以使用之前过程中的策略进行训练，用于生成采样数据序列的策略和用于实际决策的待评估和改进策略是不同的，即生成的数据“离开”了待优化的策略锁决定的决策序列轨迹。
   on-policy和off-policy只是训练方式的界限，在有时一个算法甚至可能有on-policy和off-policy的不同实现，理解概念即可。

Q4: 什么是online training和offline training？我们通常如何实现offline training？
 - Answer： Offline training即是training时不使用collector与环境进行交互，而是直接使用fixed dataset作为算法的输入， 比如behavior cloning就是经典的Offline training算法。 我们通常使用batch为单位将fixed dataset输入，因此offline RL又称Batch RL。


Q5: 什么是expolration and expolitation？我们通常使用哪些方法平衡expolration and expolitation？
 - Answer：Expolration即是RL中的agent需要不断的去探索环境的不同状态，而Expolitation则是agent需要去选择当前状态下尽可能的收益高的动作。
   平衡expolration and expolitation有很多种方式，在不同的算法中有不同的实现，比如可以采用一定概率选择随机动作，或者在动作选择时加入一定噪声等方式。


Q6: 什么是discrete space和continuous space？我们哪些算法适用于discrete space？哪些算法适用于continuous space？
 - Answer：discrete space就是环境的动作空间离散，比如玩石头剪刀布时我们的动作就是离散的三种动作。continuous space环境的动作空间连续，比如我们在开车的时候控制方向盘的角度，或者机械臂在抓取过程中各个关节的控制，就是连续的动作。

Q7: 为什么要使用replay buffer？experience replay作用在哪里？
 - Answer：通过使用replay buffer我们可以将experience存入buffer，而在之后的训练中取出buffer中的experience使用。经验回放技术（experience replay）就是将系统探索环境获得的样本保存起来，然后从中采样出样本以更新模型参数。

Q8: 算法中的value(state function), Q值(state-action function)和advantage分别是什么意思？
 - Answer：
   Value即是算法中的 :math:`V(S_t)`， 代表某时刻某个状态下的状态价值函数，即某个策略经过该状态之后预计能得到的reward数值。
   Q值即是算法中的 :math:`Q(S_t, A_t）`，代表某时刻某个状态下选择了某个动作后的状态动作价值函数，经过该状态说选择某个动作之后预计能得到的reward数值。
   Advantage则是与动作相关的 :math:`A(S_t, A_t) = Q(S_t, A_t) - V(S_t)`， 代表某时刻某个状态下选择了某个动作相比与选择其他动作的优势，预计比选择其他动作之后能多获得多少reward数值。
