Survey On Exploration
===============================

.. toctree::
   :maxdepth: 7

0. Preparation Facts
***********************************
    - Terms:
        - Exploration bonus
            - 定义Exploration增益的常用方式之一，定义generic的Bonus Function，使其根据不同的exploration增益依据适应到不同的RL算法中。(`DRL MDPs Overview <./images/RL_survey_2020.png>`_)

            - 如在常见的count-based approach中，Novelty Function基于对某个state的熟悉度通过转化形成对Reward增益的Bonus Function。
                - 一些Bonus Def
                    ..  image:: ./images/Diff_Bonus.png
        -  `Thompsom Sampling <https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf>`_
            -  可以用Policy-basedDRL得到样本训神经网络更新policy的思想理解Thompsom Sampling：
                -  posterior distribution : prior + sampling
  
        -  `UCB <https://www.geeksforgeeks.org/upper-confidence-bound-algorithm-in-reinforcement-learning/>`_ 
            -  一种设计Bonus的方法： 基于visitation 的频率来增益reward， 属于count based approach中常用的实现exploraion的方法。在AlphaGo MCTS算法中有使用。
                .. image:: ./images/UCB.png
        -  `Boltzmann distribution <https://www.mikulskibartosz.name/using-boltzmann-distribution-as-exploration-policy-in-tensorflow-agent/>`_
            -  源于统计力学，“理想气体”分子间基本没有作用力情况下的分布。在Soft-Q Learning ，在使用stochastic policy的基础上对multimodal，为了更好的收敛Q函数收敛，使用energy-based policy
                .. image:: ./images/energy-based-policy.png
                .. image:: ./images/equation.svg
                .. image:: ./images/equation2.svg 
                (这也是最大熵RL的optimal policy最优策略的形式)
                这样的policy能够为每一个action赋值一个特定的概率符合Q值的分布，也就满足了stochastic policy的需求。


                

        -  `SimHash <https://www.cnblogs.com/sddai/p/10088007.html>`_
            -  使用Hash进行伪计数的count based方法，SimHash 属于 locality-sensitive hashing（LSH），在visitation基础上体现了包含local vs not local层面的exploration approach。
                .. image:: ./images/count-hashing-exploration.png
        -  `Info Gain <https://victorzhou.com/blog/information-gain/>`_
            -  理解Information-theoretic exploration即为state += agent.info，且定义信息增益形成对Reward增益的。代表算法如：
                -  `VIME（信息最大化探索) <https://arxiv.org/abs/1605.09674>`_
                    -  Good for sparse-reward exploration problems
                -  `ICM （定义curiosity，实现上是transition dist entropy的近似，使内在奖励函数优化的探索) <https://pathak22.github.io/noreward-rl/>`_
                    -  Error -> Reward
        -  `RND （略区别于以上VIME ICM的intrinsic reward设计，使用nn预测误差表示Novelty并与extrinstic结合) <https://arxiv.org/abs/1810.12894>`_
                    -  Bonus = the error of a neural network predicting features of the observations given by a fixed randomly initialized neural network
                    -  Flexibly combine intrinsic and extrinsic rewards
        -  `Contextural Bandit <./images/CB.png>`_; `Bayesian RL <https://cs.uwaterloo.ca/~ppoupart/ICML-07-tutorial-slides/ICML-07-Tutorial-Slides.html>`_; `PAC <https://www.cs.cmu.edu/~mgormley/courses/10601-s17/slides/lecture28-pac.pdf>`_; `POMDP <https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process>`_
    -  Goals:
            1. Understand & Motivate exploration 
            2. Understand practive of exploration: How to (2.1) derive exploration methods from perspectives to Formula/Model, and (2.2) how to practive in DRL.
            3. Understand different envs for researching exploration problems.
    

            
1. Motivation on Exploration 
***********************************
    - Why Exploration? See Example:`Montezuma's Revenge <./images/mr.png>`_ 
        - Recalling on evaluating performance
            - Defining Reward (func)
            - Aternative - mimic expert i.e. behavior cloning 
                - Difficulties:
                    - Capability difficulties
                    - Identifying salient parts' difficulties
            - Objective - Reason about what the expert is trying to achieve
        - Challenges (hopefuly to be solved by exploration)
            - Decide on (Definition / Form of) Reward : How to get strategies without instant reward but big final rewards
            - Exploration vs Exploitation : How to decide condition for attempting new behaviors 
        - Exploration Problems 不过分忽略 AND 不过分关注
            - Hard-Exploration i.e. exploration in an environment with very sparse or deceptive rewards
            - The Noisy TV Problem : Agent gets reward by a noise (random uncontrollable and unpredictable reward consistently, but fails to proceed to any meaningful progress" "怎么让该专心走迷宫的智能体不分心看电视不走了？"
                .. image:: ./images/the-noisy-TV-problem.gif
        - Limitation (on `Tractability and Optimization <./images/tract.png>`_ ) 
            .. image:: ./images/tract.png


2. Measurements on Exploration and Integrating into Reward i.e. the "How to?"
********************************************************************************
    - Classic Approachs
        - Statistical Approach:
            - Epsilon-greedy; UCB; Boltzmann exploration; Thompson sampling
    - More Perspectives and Corresponding Methods (WIP)
        - Local vs Not Local
        - Common reward manipulation method:
            - Optimism-based exploration i.e. 'New' == 'Good', typically defining a novelty term (UCB) which integrate into bonus (The naive idea is simply add as a bonus to the reward, no need of tuning)
                - Count based / Error Based : count on visitations of states
                    - Naive (Problem: sometimes we never see a state twice. )
                    - Hashing (SimHash, to solve the problem that some states are similar to each other)
                  
                - Prediction Based : stores all the experiences encountered by the robot, estimate novelty with the prediction error 
                    -  Forward dynamics prediction model i.e.  `Intelligent Adaptive Curiosity <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.177.7661&rep=rep1&type=pdf>`_
                        .. image:: ./images/IAC.png

                    -  Intrinsic Curiosity Module i.e. `ICM <https://arxiv.org/abs/1705.05363>`_
                        .. image:: ./images/ICM.png
                - Memory Based : 
                    - Episodic Curiosity `(EC) <https://arxiv.org/abs/1810.02274>`_
                         .. image:: ./images/EC.png
            - Thompson sampling style Exploration e.g. Bootstrapped DQN 
                - Q-value Based: learn distribution over Q-functions or Policy 
                - sample and act according to sample
            - Information-theoretic exploration
                - Add an entropy term H(π(a|s)) into the loss function, encouraging the policy to take diverse actions.
            - Noise Approach:
                - Add noise into the observation, action or even parameter space
            - Direct Exploration:
                - Phase 1 "Go Explore" + Phase 2 "Backward Algorithm"  `(Go-Explore) <https://arxiv.org/abs/2004.12919>`_
                    
                    .. image:: ./images/policy-based-Go-Explore.png


3. Practices of Algo and Env (WIP.)
***********************************
    - Curiosity Algorithm (Optimism-based exploration)
        - CDP （`Curiosity-Driven Experience Prioritization via Density Estimation <http://arxiv.org/pdf/1902.08039v2.pdf>`_）
            - Overview 
                - 好奇心驱动优先排序（CDP）框架, 希望平衡地探索memory buffer里的样本
                - 在（Agent 探索 -> 轨迹收集2Buffer -> 学习）过程中，focus on ucommon events and their tracks to encourage the agent to over-sample those trajectories that have rare achieved goal states
            - Algorithm 
            .. image:: ./images/CDP.png
            - Representation & Manipulation of Exploration
                - Combined CDP with Deep Deterministic Policy Gradient (DDPG) with or without Hindsight Experience Replay (HER).
            - Environment
                - OpenAI gym
                - 6 robot manipulation task
        - BeBold (`Exploration Beyond the Boundary of Explored Regions <https://yuandong-tian.com/BeBold.mp4>`_
            - `Overview <https://zhuanlan.zhihu.com/p/337759337>`_ (<- 作者写的说明文档）
                - 一言以蔽之：定义新旧探索区域的边界并且鼓励边界跨越。具体实现为给定一条智能体走过的路径，估计各状态 s的访问次数 N（s） ，然后把它们的倒数相减，再做一个ReLU截断，就是BeBold。不仅希望像Count-based或者是RND那样去探索那些访问次数[公式]较少的状态，更希望能探索在充分访问区域与未充分访问区域交邻的边界.
            .. image:: ./images/bebold.jpg
            - Algorithm
            .. image:: ./images/BeBold.png
            - Representation & Manipulation of Exploration
                - 探索在充分访问区域与未充分访问区域交邻的边界
                .. image:: ./images/bb_manipulation.jpg
                - 为避免重复游走增加每局访问数（episodic visitation count）的限制，让每局每个状态最多能拿到一次奖励。
                .. image:: ./images/bb_m2.jpg
                - 用的是Random Network Distillation（RND）的方法，用一个预测网络（predictor network）去拟合另一个固定的随机神经目标网络（random fixed target network）的输出。一个状态 s 被访问的次数越多，则预测网络和目标网络的差值就越小。用两者的差值，就可以反向估计出一个状态的访问次数,即为Novelty function N（s) 
                .. image:: ./images/ablation_bb.jpg   
            -  Environment
                - Mini Grid
  
    -  Q-value Exploration 
   
    -  Direct Exploration 
        - Go Explore `Paper1 <https://arxiv.org/abs/1901.10995>`_ `Paper2 <https://arxiv.org/pdf/2004.12919.pdf>`_
            - Overview
                - GE算法区别于之前的探索强化学习算法最大的特点即是把returning 和 exploring 更大限度的分开——先形成一个return policy再在此基础上explore。
            - Algorithm 
                - 回到下图。在return policy (Go phase 或者 phase 1）阶段在存档中guided by heuristic地选择一个和promising cells关联的state 以minize explore降低了return failure，然后在下一个阶段(explore phase 或者 phase 2）采用purely exploratory policy，即随机采样或者从另一个policy中采样.
                    - 步骤如下
                        - Probabilistically select a state from the archive, guided by heuristics that prefer states associated with promising cells.
                        - Return to the selected state, such as by restoring simulator state or by running a goal-conditioned policy. 
                        - Explore from that state by taking random actions or sampling from a policy.
                        - Map every state encountered during returning and exploring to a low-dimensional cell representation. 
                        - Add states that map to new cells to the archive and update other archive entries.
                .. image:: ./images/policy-based-Go-Explore.png
                - GE算法的一个结果是两个状态之间变化轨迹更新，如果途径状态不同那么会保存新的轨迹而不会抛弃旧的轨迹。在文章中总结为维护存档。
                .. image:: ./images/ge_detachment.png
                .. image:: ./images/ge_overview.png
                - 总结Go explore 算法Key Principal： (1) remember good exploration stepping stones, (2) first return to a state, then explore and, (3) first solve a problem, then robustify (if necessary).
            - Representation & Manipulation of Exploration
                -  总结Go explore 算法对状态处理的特点：把状态表达为低维度的cell以及domain knowledge的引入，前者的实现为对基本状态的灰度化和低像素化后存储，而后者表现为在上面提取的状态的基础上还使用编写的程序从图中提取一些与游戏相关的信息作为状态特征，比如智能体的x, y坐标，当前房间的位置，当前处于的关卡数等。
                .. image:: ./images/ge_grayscale.png
            - Results
                - 不使用 domain knowledge 的结果就超越了人类专家的水平。使用了专家知识之后，效果提升。
                .. image:: ./images/ge_mr_result.png
                .. image:: ./images/ge_domain.png
 
            - Environment
                - Montezuma’s Revenge 
                - Pitfall (此处不具体展示）
     


4. Integration with DRL methods 
***********************************
    - ICM
        - & DQN
        - & PPO

5. Environments
***********************************
    - Can pip
        - Toy Version Montezuma's Revenge 
        - `MontezumaRevenge-v0 <https://gym.openai.com/envs/MontezumaRevenge-v0/>`_
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
  