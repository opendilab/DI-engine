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
