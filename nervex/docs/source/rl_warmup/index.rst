.. role:: math
   :format: html latex
..

强化学习warm up
===============

导论
----

机器学习与强化学习
 -  强化学习目前与机器学习有很大关联性，但是不同于机器学习
 -  强化学习有很多与机器学习无关的基础方法，如Q-learning等
 -  强化学习研究在交互中学习的\ **计算性**\ 方法。
 -  强化学习侧重于以交互目标为导向进行学习

强化学习特征
 - 侧重在线学习
 - 延迟收益 
 - 平衡\ **试错与开发** 
 - 从整个过程考虑目标导向的智能体与不确定环境的交互

监督学习与强化学习
 - 有监督学习

   - 从外部监督者提供的带标注训练集中进行学习 
 - 无监督学习

   - 使用无标注训练集中进行学习，寻找未标注数据中的隐含结构

强化学习的关键要素 
 - 策略 
 - 收益信号 
 - 价值函数 
 - 模型(Optional)

强化学习的历史发展
 - 试错法
 - 最优控制
 - 时许差分方法

Q&A
~~~

Q1:什么是强化学习？
 -  强化学习是智能体（Agent）以“试错”的方式进行学习
 -  “学习做什么（将情景映射为动作）才能使得数值化的收益信号（reward）最大化”

基本概念
--------

试错与开发- 多臂赌博机（Multi-Arm Bandit）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: https://pic2.zhimg.com/80/v2-de3eeb3c1136b278f72df468078f97dc_720w.jpg?source=1940ef5c
   :alt: img


什么叫bandit? 赌场的赌博机有个外号叫单臂强盗（single-armed BANDIT），因为即使老虎机只有一个摇臂，也会抢走你口袋里的钱。

multi-arm bandit
^^^^^^^^^^^^^^^^

一家赌场有 :math:`K` 台老虎机，假设每台老虎机都有一定概率(:math:`p_i`)吐出一块钱，有一定概率(\ :math:`1-p_i`)不吐钱。现在你无法提前知道老虎机的概率分布，只能通过使用老虎机的结果去推测概率。但是你一共只能摇
:math:`T` 次老虎机，在这种情况下，使用什么样的策略才能使得你摇出更多的钱呢？

-  去试一试每个赌博机，并且要有策略的去试

   -  想要获得各个老虎机的吐钱概率：试验各个老虎机，Exploration
   -  想要获得最大的收益：多摇那个吐钱概率高的老虎机，Exploitation

-  bandit算法就是为了平衡\ **Exploration and Exploitation**
-  bandit算法应用广泛，比如最直接的应用就是推荐系统，还可应用于MDP模型的游戏、等等。

经典的bandit算法:
^^^^^^^^^^^^^^^^

朴素bandit
''''''''''''

-  先随机试若干次，然后一直选最大的那个

Epsilon-Greedy
''''''''''''''

-  :math:`\epsilon`\ 概率随机选择做exploration，\ :math:`1-\epsilon`\ 概率选择当前平均收益最大的臂做exploitation
-  比较常见的改进是通过控制\ :math:`\epsilon`\ 值来平衡exploration和exploitation
-  Epsilon-Greedy曲线

Thompson sampling
'''''''''''''''''

-  假设每个臂是否产生收益，其背后有一个概率分布，产生收益的概率为:math:`p`，同时该概率 :math:`p`的概率分布符合 :math:`Beta(wins,lose)`
   分布，每个臂都维护一个 :math:`Beta` 分布。每次试验后，选中一个臂摇一下，有收益则该臂的 :math:`wins` 增加 1
   ，否则该臂的 :math:`lose` 增加 1。
-  beta分布介绍：
    https://www.zhihu.com/question/30269898
-  选择方式：用每个臂现有的Beta分布产生随机数，选择随机数中最大的那个臂
-  Thompson
   bandit算法的本质是后验证采样，性能与最好的无分布方法(UCB)相似

UCB（Upper Confidence Bound）
'''''''''''''''''''''''''''''

-  对每一个臂都试一遍，之后在任意时刻 :math:`t` 
   按照如下公式计算每个臂的分数，然后选择分数最大的臂
:math:`x_j(t) + \sqrt{\frac{2lnt}{T_{j,t}}}`

   其中 :math:`j` 为编号即表示第j臂， :math:`T_{j,t}` 为在t时刻第j个臂累计的被使用次数。
-  UCB在简单的bandit算法中，是相对效果最好的一个
..
 TODO：对比用ipy文件


马尔可夫决策过程(MDP)
~~~~~~~~~~~~~~~~~~~
马尔可夫决策过程(MDP)是强化学习在数学上的理想化形式，也是最常见的常见模型。
这个问题的数学化结构中有若干关键要素

- 回报(reward)
- 价值函数(value function)
- 贝尔曼方程(Bellman function)
通过MDP过程，就能大致理解深度学习智能体环境交互的定义。

.. note::
    https://zhuanlan.zhihu.com/p/28084942


动态规划(DP)
~~~~~~~~~~~
动态规划DP是一类优化方法，在给定一个MDP完备欢迎的情况下，可以计算最优的策略。但是对于强化学习问题，传统DP的作用十分有限。
很多强化学习问题无法获得完备的环境模型，且DP在大维度时计算复杂度极高。不过DP仍不失为一个重要理论，很多其他方法都是对DP的一种近似，
只不过降低了计算复杂的和对环境模型完备的假设。

策略迭代与价值迭代
 - 策略迭代

   - 策略评估
   - 策略更新
 - 价值迭代

异步DP
 - 不使用系统遍历状态集的形式来组织算法

广义策略迭代GPI

DP的效率问题
 - 维度灾难


蒙特卡洛方法(MC)
~~~~~~~~~~~~~~

- 首次访问MC
- 每次访问MC 

同轨策略与离轨策略

- 同轨策略(on policy)

  - 用于生成采样数据序列的策略和用于实际决策的待评估和改进策略是相同的
- 离轨策略(off policy)

  - 用于生成采样数据序列的策略和用于实际决策的待评估和改进策略是不同的，即生成的数据“离开”了待优化的策略锁决定的决策序列轨迹

蒙特卡洛方法对比DP的优势

- 不需要描述环境的模型
- 可以使用数据仿真和采样模型
- 可以只评估关注的状态
- 在马尔可夫性不成立时性能损失较小





时序差分学习(TD)
~~~~~~~~~~~~~~

..

    n步自举法
    ~~~~~~~~~



    Q&A
    ~~~

    Q1:

    -

    算法
    ----

    RL Algorithm
    ~~~~~~~~~~~~

    Paper List
    ^^^^^^^^^^

    Q&A
    ^^^

    MARL
    ~~~~

    Paper List
    ^^^^^^^^^^

    Q&A
    ^^^

    Large Scale RL Training
    ~~~~~~~~~~~~~~~~~~~~~~~

    Paper List
    ^^^^^^^^^^

    Q&A
    ^^^


    .. |img| image:: https://bkimg.cdn.bcebos.com/formula/6b72394d178730e1676d40f3824c2f46.svg


RL Warmup
===============================

.. toctree::
   :maxdepth: 3


RL Algorithm
------------

Paper List
~~~~~~~~~~
1. DQN
2. Dueling DQN
3. Prioritized Replay Buffer
4. A2C
5. PPO
6. GAE
7. DDPG
8. SAC

.. note::
    一个部分论文的链接: `传送门 <https://zhuanlan.zhihu.com/p/23600620>`_

Blog List
~~~~~~~~~~
1. `强化学习入门简述 <https://zhuanlan.zhihu.com/p/64197895?utm_source=wechat_session&utm_medium=social&utm_oi=778950235199127552&utm_content=sec>`_
2. `强化学习之遇到的一些面试问题 <https://zhuanlan.zhihu.com/p/52143798?utm_source=wechat_session&utm_medium=social&utm_oi=778950235199127552&utm_content=sec>`_
3. `炼丹感悟：On the Generalization of RL <https://zhuanlan.zhihu.com/p/105898705?utm_source=wechat_session&utm_medium=social&utm_oi=778950235199127552&utm_content=sec>`_
4. `Pytorch RL tutorial <https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html>`_

Questions(即需要理解清楚的概念和问题)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. model-free and model-based
2. value-based, policy-based, actor-critic
3. on-policy and off-policy
4. off-policy and offline training(batch RL)
5. expolration and expolitation
6. discrete space and continuous space
7. replay buffer
8. td-error(temporal difference), 1-step, n-step, MC, DP
9. value(state function), Q(state-action function), advantage
10. return and reward

MARL
------
to be continued

Large Scale RL Training
-----------------------


Paper List
~~~~~~~~~~
1. A3C
2. Ape-X
3. IMPALA
4. Seed RL
5. ACME
6. AlphaGo
7. AlphaStar
8. OpenAI Five
9. Rllib

Blog List
~~~~~~~~~
1. `最前沿：深度强化学习的强者之路 <https://zhuanlan.zhihu.com/p/161548181?utm_source=wechat_session&utm_medium=social&utm_oi=30146627108864&utm_content=first&from=singlemessage&isappinstalled=0&wechatShare=1&s_r=0>`_


Questions(即需要理解清楚的概念和问题)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. async training in A3C(gradient)
2. Actor-Learner Architecture
3. v-trace(importance weight)
4. MCTS(AlphaGo)
5. League(AlphaStar)


.. note::
    以上包含内容精读食用最佳，不宜囫囵吞枣，最好结合相关具体代码实现
