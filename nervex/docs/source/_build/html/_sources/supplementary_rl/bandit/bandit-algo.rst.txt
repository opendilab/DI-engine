Bandit Concepts
^^^^^^^^^^^^^^^

Basic Settings
--------------

.. figure:: https://pic2.zhimg.com/80/v2-de3eeb3c1136b278f72df468078f97dc_720w.jpg?source=1940ef5c
   :alt: img


什么叫bandit? 赌场的赌博机有个外号叫单臂强盗（single-armed BANDIT），因为即使老虎机只有一个摇臂，也会抢走你口袋里的钱。
我们用 :math:`A` 这个集合表示action space，也就是arm的集合。只考虑最基本的情形，因此我们让 :math:`A` 只包含有限个元素（记元素个数为 :math:`K` ，即一共有 :math:`K` 个arm）。同样，只考虑离散时间的决策问题，我们认为我们一共有 :math:`T` 个阶段进行决策， :math:`T` 事先已知。那么在 :math:`t=1, ..., T` 的每个阶段中，一个bandit算法应该做如下两件事：

- 算法从 :math:`A` 中选择一个arm :math:`a_t`  

- 算法观察到 :math:`a_t` 这个arm返回的奖励 (reward) :math:`r_t`

这是最基本的RL setting，我们的算法每次选择一个action，所得到的feedback即我们的reward。IID的setting即是说每个arm返回的reward都是独立于彼此，且分布在整个时间轴上不变。


Multi-Arm Bandit
----------------

一家赌场有 :math:`K` 台老虎机，假设每台老虎机都有一定概率(:math:`p_i`)吐出一块钱，有一定概率(\ :math:`1-p_i`)不吐钱。现在你无法提前知道老虎机的概率分布，只能通过使用老虎机的结果去推测概率。但是你一共只能摇:math:`T` 次老虎机，在这种情况下，使用什么样的策略才能使得你摇出更多的钱呢？

-  去试一试每个赌博机，并且要有策略的去试

   -  想要获得各个老虎机的吐钱概率：试验各个老虎机，Exploration
   -  想要获得最大的收益：多摇那个吐钱概率高的老虎机，Exploitation

-  bandit算法就是为了平衡\ **Exploration and Exploitation**
-  bandit算法应用广泛，比如最直接的应用就是推荐系统，还可应用于MDP模型的游戏、等等。


Bandit Algorithms
-----------------

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

-  假设每个臂是否产生收益，其背后有一个概率分布，产生收益的概率为 :math:`p`，同时该概率 :math:`p`的概率分布符合 :math:`Beta(wins,lose)`
   分布，每个臂都维护一个 :math:`Beta` 分布。每次试验后，选中一个臂摇一下，有收益则该臂的 :math:`wins` 增加 1，否则该臂的 :math:`lose` 增加 1。
-  beta分布介绍：https://www.zhihu.com/question/30269898
-  选择方式：用每个臂现有的Beta分布产生随机数，选择随机数中最大的那个臂
-  Thompson
   bandit算法的本质是后验证采样，性能与最好的无分布方法(UCB)相似

UCB（Upper Confidence Bound）
'''''''''''''''''''''''''''''

-  对每一个臂都试一遍，之后在任意时刻 :math:`t` 
   按照如下公式计算每个臂的分数，然后选择分数最大的臂

   :math:`x_j(t) + \sqrt{\frac{2lnt}{T_{j,t}}}`

   其中 :math:`j` 为编号即表示第j臂， :math:`T_{j,t}` 为在t时刻第j个臂累计的被使用次数。

-  UCB算法虽然简单，但其在实际应用中往往能取得相对好的效果，可见下文中的bandit example。

-  UCB的效果是有理论保障的。UCB的累计regret为 :math:`O(Knlog(n))` ，即其在选用了n次arm之后产生的regret为 :math:`O(log(n))` 的。在2002年提出UCB的论文 `Using Confidence Bounds for Exploitation-Exploration Trade-offs <https://www.jmlr.org/papers/volume3/auer02a/auer02a.pdf>`_ 中已有了证明。
