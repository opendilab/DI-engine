RL Exploration
~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 7

0. Preparation Facts
***********************************
    - Terms：
        - Exploration bonus
            - 定义Exploration增益的常用方式之一，定义generic的Bonus Function，使其根据不同的exploration增益依据, 适应到不同的RL算法中。(`DRL MDPs Overview <images/RL_survey_2020.png>`_)

            - 如在常见的count-based approach中，Novelty Function基于对某个state的熟悉度通过转化形成对Reward增益的Bonus Function。
                - 一些Bonus的定义例子
                    ..  image:: images/Diff_Bonus.png
        -  `Thompson Sampling <https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf>`_
            -  TS 可使用先验经验的信息，实现TS重复的流程如下：
                -  设定先验分布：为每一个可能的选项建立一个 Distribution （如选择Beta则先验经验体现在 α与β的初值设定）
                -  采样：使每个贝塔分布产生一个随机数值，选择结果最大的作为本次选项
                -  调参：根据实验根据实验结果调整参数（α与β）
            -  因而Policy-basedDRL得到样本，训神经网络更新policy的思想，可以用来理解Thompsom Sampling：
                -  TS: prior + sampling ==> posterior distribution
        -  Stochastic policy vs. Deterministic policy
            -  Stochastic policy
                -  从state set S​到action set A​的条件概率分布, 在这个S上所有的action都存在一个被选到的概率。在stochastic policy中，action来自policy distribution's sampling.
            -  Deterministic policy
                -  S->A的映射。一个state s到对应a​，因policy固定，每一对有映射关系s得到确定的a（概率始终为1）。

        - `Multimodal <https://github.com/pliang279/awesome-multimodal-ml#survey-papers>`_
            - 引入多个模态的数据来增强强化学习模型的性能。e.g.智能体捕捉到的环境的声音与的环境图像就可看作是环境信息的两个模态的数据，这两个不同的环境数据来源可以用以提高模型性能。

       
        -  `UCB <https://www.geeksforgeeks.org/upper-confidence-bound-algorithm-in-reinforcement-learning/>`_ 
            -  一种设计Bonus的方法： 基于visitation 的频率来增益reward， 属于count based approach中常用的实现exploraion的方法。在AlphaGo MCTS算法中有使用。
                .. image:: images/UCB.png
        -  `Boltzmann distribution <https://www.mikulskibartosz.name/using-boltzmann-distribution-as-exploration-policy-in-tensorflow-agent/>`_
            -  定义
                -  源于统计力学，“理想气体”分子间基本没有作用力情况下的分布。
            -  使用
                -  在Soft-Q Learning ，在使用stochastic policy的基础上对multimodal，为了更好的收敛Q函数收敛，使用energy-based policy
                .. note:: Energy-based policy 即熵策略，一种基于能量的表达策略。 在热力学和信息论中，熵为随机程度的度量，因此被用来参与对探索奖励的计算 e.g.最大熵RL
                .. image:: images/energy-based-policy.png
                .. image:: images/equation.svg
                .. image:: images/equation2.svg 
                (这也是最大熵RL的optimal policy最优策略的形式)
                这样的policy能够为每一个action赋值一个特定的概率符合Q值的分布，也就满足了stochastic policy的需求。


                

        -  `SimHash <https://www.cnblogs.com/sddai/p/10088007.html>`_
            -  使用Hash进行伪计数（一种先验计数方法，可以理解为观测没发生时可能性的动态变化）的count based方法，SimHash 属于 locality-sensitive hashing（LSH），在visitation基础上体现了包含相似程度层面的exploration approach。
                .. image:: images/count-hashing-exploration.png
                .. note:: 为什么需要伪计数（Pseudo-Count) 呢？首先理解count-based的探索基于state的频率计算探索奖励，但是在如星际的环境中几乎不存在相同state（即所有state count都是1或0）。因此我们需要伪计数来近似，从而把足够相似的环境认为是可以增加count的。

        -  `Info Gain <https://victorzhou.com/blog/information-gain/>`_
            -  理解Information-theoretic exploration即为state += agent.info，且定义信息增益形成对Reward增益的。代表算法如：
                -  `VIME（信息最大化探索) <https://arxiv.org/abs/1605.09674>`_
                    -  Good for sparse-reward exploration problems
                -  `ICM （定义curiosity，实现上是transition dist entropy的近似，使内在奖励函数优化的探索) <https://pathak22.github.io/noreward-rl/>`_
                    -  Error -> Reward
        -  `RND （略区别于以上VIME ICM的intrinsic reward设计，使用nn预测误差表示Novelty并与extrinstic结合) <https://arxiv.org/abs/1810.12894>`_ （详细介绍见 pt3 error based部分）
                    -  Bonus = the error of a neural network predicting features of the observations given by a fixed randomly initialized neural network
                    -  Flexibly combine intrinsic and extrinsic rewards
        -  `Contextural Bandit <images/CB.png>`_; `Bayesian RL <https://cs.uwaterloo.ca/~ppoupart/ICML-07-tutorial-slides/ICML-07-Tutorial-Slides.html>`_; `PAC <https://www.cs.cmu.edu/~mgormley/courses/10601-s17/slides/lecture28-pac.pdf>`_; `POMDP <https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process>`_
    -  Goals:
            1. Understand & Motivate exploration 
            2. Understand practive of exploration: How to (2.1) derive exploration methods from perspectives to Formula/Model, and (2.2) how to practive in DRL.
            3. Understand different envs for researching exploration problems.
    

            
1. Motivation on Exploration 
***********************************
    - Why Exploration? See Example:`Montezuma's Revenge <images/mr.png>`_ 
        - Recalling on evaluating performance
            - Defining Reward (func)
            - Alternative - mimic expert i.e. behavior cloning 
                - Difficulties:
                    - Capability difficulties
                    - Identifying salient parts' difficulties
            - Objective - Reason about what the expert is trying to achieve
        - Challenges (hopefuly to be solved by exploration)
            - Decide on (Definition / Form of) Reward : How to get strategies without instant reward but big final rewards
            - Exploration vs Exploitation : How to decide condition for attempting new behaviors 
        - Exploration Problems 不过分忽略 AND 不过分关注
            - Hard-Exploration i.e. exploration in an environment with very sparse or deceptive rewards
            - The Noisy TV Problem : Agent gets reward by a noise (random uncontrollable and unpredictable reward consistently, but fails to proceed to any meaningful progress" "怎么让智能体专心走迷宫而不是分心地看电视不走了？"
                .. image:: images/the-noisy-TV-problem.gif
        - Limitation (on `Tractability and Optimization <images/tract.png>`_ ) 
            .. image:: images/tract.png


2. Measurements on Exploration and Integrating into Reward i.e. the "How to?"
***********************************
到这里我们已经知道使用探索增益的想要解决的两大问题是什么：如何有效定义{稀疏,复杂,有欺骗性的}奖励 + 摆脱局部最优的困境。接下来进入方法的探讨。
    - Classic Approachs
        - Statistical Approach:
            - Epsilon-greedy; UCB; Boltzmann exploration; Thompson sampling
    - More Perspectives and Corresponding Methods (WIP)
        - Common reward manipulation methods:
            - Optimism-based exploration i.e. 'New' == 'Good', typically defining a novelty term (such as UCB) which integrate into bonus (The naive idea is simply to add a bonus to the reward, no need of tuning)
                - Count based / Error Based : count on visitations of states
                    - Naive (Problem: sometimes we never see a state twice. )
                    - Hashing (e.g. SimHash, to solve the problem that some states are similar to each other)
                  
                - Prediction Based : stores all the experiences encountered by the robot, estimate novelty with the prediction error 
                    -  Forward dynamics prediction model i.e.  `Intelligent Adaptive Curiosity <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.177.7661&rep=rep1&type=pdf>`_
                        .. image:: images/IAC.png

                    -  Intrinsic Curiosity Module i.e. `ICM <https://arxiv.org/abs/1705.05363>`_
                        .. image:: images/ICM.png
                - Memory Based : 
                    - Episodic Curiosity `(EC) <https://arxiv.org/abs/1810.02274>`_
                         .. image:: images/EC.png
            - Thompson sampling style Exploration e.g. Bootstrapped DQN 
                - Q-value Based: learn distribution over Q-functions or Policy 
                - sample and act according to sample
            - Information-theoretic exploration
                - Add an entropy term H(π(a|s)) into the loss function, encouraging the policy to take diverse actions.
            - Noise Approach:
                - Add noise into the observation, action or even parameter space
            - Direct Exploration:
                - Phase 1 "Go Explore" + Phase 2 "Backward Algorithm"  `(Go-Explore) <https://arxiv.org/abs/2004.12919>`_
                 .. image:: images/policy-based-Go-Explore.png


3. Practices of Algo and Env (WIP.)
***********************************
    - Curiosity Algorithm (Optimism-based exploration)
        - CDP （`Curiosity-Driven Experience Prioritization via Density Estimation <http://arxiv.org/pdf/1902.08039v2.pdf>`_）
            - Overview 
                - 好奇心驱动优先排序（CDP）框架, 希望平衡地探索memory buffer里的样本
                - 在（Agent 探索 -> 轨迹收集2Buffer -> 学习）过程中，focus on ucommon events and their tracks to encourage the agent to over-sample those trajectories that have rare achieved goal states
            - Algorithm 
            .. image:: images/CDP.png
            - Representation & Manipulation of Exploration
                - Combined CDP with Deep Deterministic Policy Gradient (DDPG) with or without Hindsight Experience Replay (HER).
            - Environment
                - OpenAI gym
                - 6 robot manipulation task
        - BeBold (`Exploration Beyond the Boundary of Explored Regions <https://yuandong-tian.com/BeBold.mp4>`_）
            - `Overview <https://zhuanlan.zhihu.com/p/337759337>`_ (<- 作者写的说明文档）
                - 一言以蔽之：定义新旧探索区域的边界并且鼓励边界跨越。具体实现为给定一条智能体走过的路径，估计各状态 s的访问次数 N（s） ，然后把它们的倒数相减，再做一个ReLU截断，就是BeBold。不仅希望像Count-based或者是RND那样去探索那些访问次数[公式]较少的状态，更希望能探索在充分访问区域与未充分访问区域交邻的边界.
                    .. note:: ReLU: 普通的ReLU自变量负数段斜率是0，在正数段原样或按正比例输出
            .. image:: images/bebold.jpg
            - Algorithm
            .. image:: images/BeBold.png
            - Representation & Manipulation of Exploration
                - 探索在充分访问区域与未充分访问区域交邻的边界
                .. image:: images/bb_manipulation.jpg
                - 为避免重复游走增加每局访问数（episodic visitation count）的限制，让每局每个状态最多能拿到一次奖励。
                .. image:: images/bb_m2.jpg
                - 用的是Random Network Distillation（RND）的方法，用一个预测网络（predictor network）去拟合另一个固定的随机神经目标网络（random fixed target network）的输出。一个状态 s 被访问的次数越多，则预测网络和目标网络的差值就越小。用两者的差值，就可以反向估计出一个状态的访问次数,即为Novelty function N（s) 
                .. image:: images/ablation_bb.jpg   
            -  Environment
                - Mini Grid
    - Error Based Algorithm
        - RND `blog <https://openai.com/blog/reinforcement-learning-with-prediction-based-rewards/>`_ 
            - Overview
                - 直觉理解即是越可以预知的model的已知来源于更高频率的访问
                    - 而这样的model误差越小， the agent’s predictions of the output of a randomly initialized neural network will be less accurate in novel states than in states the agent visited frequently 因此RND用来增益Bonus的Novelty为 the error of predictive models
                        -  Trick 1. 游戏结束持续奖励 Continued reward 
                            - Intrinstic return does not goes to zero when episode ends. （IR ER分开训练，有灵活度选择不同的discount factor）
                            .. note:: 为什么要IR在episode结束时持续奖励，不持续奖励会带来什么问题? 对intrinsic reward来说，不持续奖励在episode结束时奖励为0，会激励agent更加保守；对于extrinsic reward，agent可能趋向于在游戏探索到的可以得到Reward的区域累计奖励然后直接gg，且定义上来说ER即是根据环境规则定义的Reward，应该在episode结束时归0。
                        -  Trick 2. Normalization 
                            - The intrinsic reward is normalized by division by a running estimate of the standard deviations of the intrinsic return.
                        
                    理论上RND 有助于解决稀疏reward空间的问题。
                       


            -  Algorithm
                -  前期分析 - Sources related to noisy TV problem OpenAI定义了一下三个prediction errors的影响来源：
                    - Prediction error is high due to：
                        - Factor 1: Novel experience
                        - Factor 2: Stochastic prediction targets
                        - Factor 3: Information necessary for the prediction is missing or target functions too complex for predictors
                   且认为Factor 1 是有助于探索的因为novelty的必要性且量化了novelty，而2和3使智能体无法区别正常reward和类noisy-TV reward，因此设计了RND w/a new exploration bonus that is based on predicting the output of a fixed and randomly initialized neural network on the next state, given the next state itself.
                -  Design (how to avoid Factor 2 and 3)
                    .. image:: images/nextstate-vs-rnd-stacked-5.svg
                    - To avoid 2，就要让神经网络给出确定性的答案，而不是给出多个答案和它们各自的可能性；
                    - To avoid 3， Target 和 Train使用同样的nn 结构
                - RND pseudo (Combined w/ PPO) 
                    - IR 的产生，对于分布式的训练架构更适合实现在什么位置？
                        - 产生：对于每个episode K的iteration都calculate intrinsic reward，并且用于optimization 当前的batch 
                        - 位置：在actor收集经验的位置实现
                    .. image:: images/RND_pseudo_code.png
                    .. note::  Prediction Network如何训练，在什么时候，用什么数据？首先，产生一个随机生成但确定的 target network 和一个被智能体收集数据集训练的a predictor network，然后对每个观测的状态计算MSE，在这个过程中，随机网络像训练网络蒸馏，而在新颖度更高的地方有更高的prediction error。
                               

                    -
            - Representation & Manipulation of Exploration （内部算法为PPO）
                - 将IR和ER标准化后相加得到奖励
                     .. image:: images/ext.svg
            - Environment
                - MontezumaRevenge
                - RND + PPO
                - `开源 <https://github.com/openai/random-network-distillation>`_


                
   
    -  Direct Exploration  approach 1. 分离return 和explore
        - Go Explore `Paper1 <https://arxiv.org/abs/1901.10995>`_ `Paper2 <https://arxiv.org/pdf/2004.12919.pdf>`_
            - Overview
                - GE算法区别于之前的探索强化学习算法最大的特点即是把returning 和 exploring 更大限度的分开——先形成一个return policy再在此基础上explore。
            - Algorithm 
                - 回到下图。在return policy (Go phase 或者 phase 1）阶段在存档中guided by heuristic地选择一个和promising cells关联的state 以minimized explore降低了return failure，然后在下一个阶段(explore phase 或者 phase 2）采用purely exploratory policy，即随机采样或者从另一个policy中采样.
                    - 步骤如下
                        - 选择state：用概率方法在state set存档中取s, guided by heuristics 以更倾向于选择和promising cells关联的s.
                        .. note:: 这里的heuristic更倾向于如何选择s呢？1. 被更少被访问过以及经验访问的return是好的 2.附近有更多未被探索过的s 3.是高level的s；
                        - 从初始状态开始前往选择的s
                            - 按照存档中的行动序列行动即可，轨迹中途经的各个状态也会被记录下来，如果要前往这些状态也可以复用前半部分的轨迹。
                        - 从当前状态出发
                            - Explore from that state by taking random actions or sampling from a policy 做K步的随机行动采样，保持大概率重复上一步。
                        - 存储
                            -  Map every state encountered during returning and exploring to a low-dimensional cell representation. 
                        - 更新
                            - Add states that map to new cells to the archive and update other archive entries.
                            .. note:: 是每次迭代都会更新吗？以下两种情况会有更新存档的必要：1.到达了新的s 2.到达了存档中的s，而走到该状态的路径更优于存档中已有的路径。因此我们知道在上一步不仅需要记录Corresponding reward on path of reaching the each state,也需要记录path的长度，用于这一步的更新。
                .. image:: images/policy-based-Go-Explore.png
                - GE算法的一个结果是两个状态之间变化轨迹更新，如果途径状态不同那么会保存新的轨迹而不会抛弃旧的轨迹。在文章中总结为维护存档。
                .. image:: images/ge_detachment.png 
                .. image:: images/ge_overview.png
                - 总结Go explore 算法Key Principal： (1) remember good exploration stepping stones, (2) first return to a state, then explore and, (3) first solve a problem, then robustify (if necessary).
            - Representation & Manipulation of Exploration
                -  总结Go explore 算法对状态处理的特点：把状态表达为低维度的cell以及domain knowledge的引入，前者的实现为对基本状态的灰度化和低像素化后存储，而后者表现为在上面提取的状态的基础上还使用编写的程序从图中提取一些与游戏相关的信息作为状态特征，比如智能体的x, y坐标，当前房间的位置，当前处于的关卡数等。
                .. image:: images/ge_grayscale.png
            - Results
                - 不使用 domain knowledge 的结果就超越了人类专家的水平。使用了专家知识之后，效果提升。
                .. image:: images/ge_mr_result.png
                .. image:: images/ge_domain.png
 
            - Environment
                - Montezuma’s Revenge 
                - Pitfall (此处不具体展示）
     
    -  Direct Exploration  approach 2. Reward-Free Exploration / 对纯粹充分探索的尝试
        -  `Reward-Free Exploration <https://arxiv.org/abs/2002.02794>`_
            -  Overview
                -  直观理解，用无奖励环境探索（在状态空间上纯探索），得到探索策略后进行采样，之后对采样的数据进行近似（对于转化矩阵）并求解.
                -  适用于对智能体行为有较理想预期的强化学习，只需考虑Reward Function的设计使智能体达到预期.
            -  Algorithm
                -  Protocol 1 i.e. Guide of Phase 1 - Exploration
                    -  a reward-free version of the MDP
                        -   the agent collects a dataset 'D' of visisted states, actions, and transitions 
                    .. image:: images/D.png
                    .. image:: images/rf_p1.png
                    第一个For Loop完成了在状态空间的无奖励探索，得到了Policy Set（使用针对每个状态的奖励函数，即每个奖励函数对于非关联状态的奖励为0，然后用内部算法计算出目标Policy Set）；第二个For Loop则是由Policy Set得到D的过程.
                -  Phase 2 - Planning
                    -  Compute Optimal Policy and measure with value function
                        -  Using reward function  r(·, ·) , compute an optimal policy using the dataset D. 
                        -  Using value function V (·; r), evaluate with # of episodes K required in the phase1 for such optimal.
                -  Algo + pseudo code of RF Explore + Plan
                    .. image:: images/rf_algo.png
                    .. image:: images/rf_exploration.png
                    .. image:: images/rf_planning.png
                    .. note:: 文章使用用 Euler 是因为该算法得到的策略集的性能和最优价值函数的大小有关，即，如果这个最优价值函数本身就不特别好（比如，即使最优策略也访问不到那些 reward 比较大但是 insignificant 的状态），那么得到的策略和最优价值函数的差距也不会特别大。 （Lemma 3.4 in Paper, also proved in Lemma 3.3 that this algorithm is applicable that it can sufficiently explore all arbitrary significant states) 而文章后来提到的MaxEnt (uniformly explore all states including insignificants) 等因为没有提出insignificant cases效果不如RF
            - Representation & Manipulation & Env（理论研究）
                .. note:: 这是一个理论研究论文，因而策略集的性能和复杂度没有实验支持。了解 reward function  r(·, ·)在phase1的状态空间的无奖励探索Loop中被用到，而value function V (·; r)是作为一个量化衡量性能的工具（不在算法内部）即可。

    -  Direct Exploration  approach 3. MaxRényi / 对纯粹充分探索的尝试
            -  `MaxRényi Exploration <https://arxiv.org/abs/2002.02794>`_
                -  `Overview <https://zhuanlan.zhihu.com/p/350594029>`_ （<-- 作者写的专栏）
                    -  基于Direct Explore第二部分RF 框架的实践
                    -  从鼓励探索上来说，以前的很多方法都采用了最大化熵的正则方法来鼓励探索。本文作者提出Renyi 熵函数（一种广义熵函数）会得到更好的探索效果。

                - Algorithm 
                    - Design （内部函数依然是PPO）
                        - 1. Adopted RF framwork
                            .. image:: images/me_framework.png
                        - 2. The Choice of objective function
                            .. image:: images/of.png
                            - （1）选择优化state-action distribution 的函数，而不只是 state distribution 的函数。原因是奖励函数一般是基于 state-action pair 定义的，因此我们希望鼓励数据集去访问各种不同的 state-action pair，而不仅仅是访问不同的状态。这里举了一个例子，说明即使最大化 state distribution 的熵函数，也可能使得某些 state-action pair 完全访问不到。
                                .. image:: images/sa.png
                            - （2） MDP只要给定一个包含所有 state-action pair 的数据集，就一定能够成功离线规划得到最优的策略。因此，目标函数的选择就有了一个完全量化可以写出的形式，即希望产生的策略能够以最少的采样次数获得这样完整的数据集。
                                .. image:: images/intractable.png
                                (然而是intractable的）Renyi 是对以上形式的近似，必要共同特征是在边缘都有类似 barrier 的结构，这样的结构能够促使智能体去更多地访问那些很难被访问到的 state-action pair.
                                .. image:: images/approx.png
                    - Algo pseudo code (w/ PPO as value function)
                        .. image:: images/ppo_renyi.png
                        
                - Representation & Manipulation & Env
                    - Exp. w/ MiniGrid - Multirooms
                    .. image:: images/multi.png
                    
                    - Exp. w/ Montezuma’s Revenge
                    .. image:: images/mr_renyi.png
                    .. image:: images/renyimr1.png



                                
                   
                    


4. Integration with DRL methods 
***********************************
    - ICM
        - & DQN
        - & PPO

5. Environments
***********************************
    - Can pip
        - Toy Version Montezuma's Revenge 
        - `MontezumaRevenge <https://gym.openai.com/envs/MontezumaRevenge-v0/>`_
        - `Gym-MiniGrid <https://github.com/maximecb/gym-minigrid>`_ 
        - `VizDoom <https://github.com/mwydmuch/ViZDoom>`_
    - `Noisy-TV (not official) <https://github.com/qqadssp/RandomMazeEnvironment>`_
    - Noisy-TV (to be found)
    - Other Atari game with sparce rewards




6. References / Remarks
***********************************
    - Resources:  
        1. `berkeley DRL17 L13 Exploration <http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_13_exploration.pdf>`_ 
        2. `lilianweng blog: Exploration Strategies in DRL <https://lilianweng.github.io/lil-log/2020/06/07/exploration-strategies-in-deep-reinforcement-learning.html>`_
        3. `Open AI 捉迷藏  <https://d4mucfpksywv.cloudfront.net/emergent-tool-use/paper/Multi_Agent_Emergence_2019.pdf>`_
  
