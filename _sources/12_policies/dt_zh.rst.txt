DT
^^^^^^^

综述
---------
如果想要将强化学习技术应用在某个决策领域，最重要的就是将原始问题转换为一个合理的 MDP （马尔科夫决策过程）问题，而一旦问题环境本身有一些不那么友好的”特性“（比如部分可观测，非平稳过程等等），常规强化学习方法的效果便可能大打折扣。另一方面，随着近些年来数据驱动范式的发展，大数据和预训练大模型在计算机视觉（Computer Vision）和自然语言处理（Natural Language Processing）领域大放异彩，比如 CLIP，DALL·E 和 GPT-3 等工作都取得了惊人的效果，序列预测技术便是其中的核心模块之一。但对于决策智能，尤其是强化学习（Reinforcement Learning），由于缺少类似 CV 和 NLP 中的大数据集和适合的预训练任务，决策大模型迟迟没有进展。

在这样的背景下，为了推进决策大模型的发展，提高相关技术的实际落地价值，许多研究者开始关注 Offline RL/Batch RL 这一子领域。具体来说，Offline RL是一种只通过离线数据集（Offline dataset）训练策略（Policy），在训练过程中不与环境交互的强化学习任务。那对于这样的任务，是否可以借鉴 CV 和 NLP 领域的一些研究成果，比如序列预测相关技术呢？

于是乎，在2021年，以 Decision Transformer[3]/Trajectory Transformer[1-2]为代表的一系列工作出现了，试图将决策问题归于序列预测，将 transformer 结构应用在RL任务上，同时与语言模型，如 GPT-x 和 BERT 等联系起来。不像传统 RL 中计算 value 函数或计算 policy 梯度， DT 通过一个屏蔽后序的 transformer 直接输出最有动作选择。通过指定期望模型达到的reward，同时借助 states 和 actions 信息，就可以给出下一动作并达到期望的 reward。 DT 的达到并超过了 SOTA model-free offline RL 算法在 Atari，D4RL (MuJoCo) 等环境上的效果。

快速了解
-------------
1. DT 是一个 **offline** 强化学习算法。

2. DT 支持 **离散（discrete）** 和 **连续（continuous）** 动作空间。

3. DT 使用 transformer 进行动作预测，但是对 self-attention 的结构进行了修改。

4. DT 的数据集结构是由算法特点决定的，在进行模型训练和测试中都要符合其要求。

重要公示/重要图示
---------------------------
DT 的结构图如下：

.. image:: images/DT.png
   :align: center


图示说明 DT 算法在进行动作 a\ :sub:`t`\  的预测时，仅与当前时间步的 r\ :sub:`t`\  和 s\ :sub:`t`\  以及之前的 r\ :sub:`t-n`\ , s\ :sub:`t-n`\ , a\ :sub:`t-n`\  相关，与之后的无关， causal transformer 就是用来实现这一效果的模块。

伪代码
---------------
.. image:: images/DT_algo.png
   :align: center

实现
----------------
DQNPolicy 的默认 config 如下所示：

.. autoclass:: ding.policy.dt.DTPolicy
   :noindex:

其中使用的神经网络接口如下所示：

.. autoclass:: ding.model.DecisionTransformer
   :members: forward
   :noindex:

实验 Benchmark
------------------
.. list-table:: Benchmark and comparison of DT algorithm
   :widths: 25 15 30 15 15
   :header-rows: 1

   * - environment
     - best mean reward (normalized)
     - evaluation results
     - config link
     - comparison
   * - | Hopper 
       | (Hopper-medium)
     - 0.753 +- 0.035
     - .. image:: images/benchmark/hopper_medium_dt.png
     - `link_2_Hopper-medium <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/hopper_medium_dt_config.py>`_
     - DT paper 
   * - | Hopper 
       | (Hopper-expert)
     - 1.170 +- 0.003
     - .. image:: images/benchmark/hopper_expert_dt.png
     - `link_2_Hopper-expert <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/hopper_medium_expert_dt_config.py>`_
     - DT paper 
   * - | Hopper 
       | (Hopper-medium-replay)
     - 0.651 +- 0.096
     - .. image:: images/benchmark/hopper_medium_replay_dt.png
     - `link_2_Hopper-medium-replay <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/hopper_medium_replay_dt_config.py>`_
     - DT paper 
   * - | Hopper 
       | (Hopper-medium-expert)
     - 1.150 +- 0.016
     - .. image:: images/benchmark/hopper_medium_expert_dt.png
     - `link_2_Hopper-medium-expert <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/hopper_medium_expert_dt_config.py>`_
     - DT paper 
   * - | Walker2d 
       | (Walker2d-medium)
     - 0.829 +- 0.020
     - .. image:: images/benchmark/walker2d_medium_dt.png
     - `link_2_Walker2d-medium <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/walker2d_medium_dt_config.py>`_
     - DT paper 
   * - | Walker2d 
       | (Walker2d-expert)
     - 1.093 +- 0.004
     - .. image:: images/benchmark/walker2d_expert_dt.png
     - `link_2_Walker2d-expert <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/walker2d_expert_dt_config.py>`_
     - DT paper 
   * - | Walker2d 
       | (Walker2d-medium-replay)
     - 0.603 +- 0.014
     - .. image:: images/benchmark/walker2d_medium_replay_dt.png
     - `link_2_Walker2d-medium-replay <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/walker2d_medium_replay_dt_config.py>`_
     - DT paper 
   * - | Walker2d 
       | (Walker2d-medium-expert)
     - 1.091 +- 0.002
     - .. image:: images/benchmark/walker2d_medium_expert_dt.png
     - `link_2_Walker2d-medium-expert <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/walker2d_medium_expert_dt_config.py>`_
     - DT paper 
   * - | HalfCheetah 
       | (HalfCheetah-medium)
     - 0.433 +- 0.0007
     - .. image:: images/benchmark/halfcheetah_medium_dt.png
     - `link_2_HalfCheetah-medium <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/halfcheetah_medium_dt_config.py>`_
     - DT paper 
   * - | HalfCheetah 
       | (HalfCheetah-expert)
     - 0.662 +- 0.057
     - .. image:: images/benchmark/halfcheetah_expert_dt.png
     - `link_2_HalfCheetah-expert <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/halfcheetah_expert_dt_config.py>`_
     - DT paper 
   * - | HalfCheetah 
       | (HalfCheetah-medium-replay)
     - 0.401 +- 0.007
     - .. image:: images/benchmark/halfcheetah_medium_replay_dt.png
     - `link_2_HalfCheetah-medium-replay <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/halfcheetah_medium_replay_dt_config.py>`_
     - DT paper 
   * - | HalfCheetah 
       | (HalfCheetah-medium-expert)
     - 0.517 +- 0.043
     - .. image:: images/benchmark/halfcheetah_medium_expert_dt.png
     - `link_2_HalfCheetah-medium-expert <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/halfcheetah_medium_expert_dt_config.py>`_
     - DT paper 
   * - | Pong 
       | (PongNoFrameskip-v4)
     - 0.956 +- 0.020
     - .. image:: images/benchmark/pong_dt.png
     - `link_2_Pong <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/qbert/qbert_dqn_config.py>`_
     - DT paper 
   * - | Breakout
       | (BreakoutNoFrameskip-v4)
     - 0.976 +- 0.190
     - .. image:: images/benchmark/breakout_dt.png
     - `link_2_Breakout <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/breakout_dt_config.py>`_
     - DT paper 

注：

以上结果是在3个不同的随机种子（即123， 213， 321）运行相同的配置得到

参考文献
----------

- Zheng, Q., Zhang, A., & Grover, A. (2022, June). Online decision transformer. In international conference on machine learning (pp. 27042-27059). PMLR.
- https://github.com/kzl/decision-transformer
