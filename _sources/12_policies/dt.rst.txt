DT (DecisionTransformer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview
---------
Applying reinforcement learning technology in a specific decision-making field necessitates the transformation of the original problem into a reasonable Markov Decision Process (MDP) problem. However, the effectiveness of conventional reinforcement learning methods may be significantly reduced if the problem environment possesses certain unfriendly characteristics, such as partial observability or non-stationary processes. On the other hand, with the development of the data-driven paradigm in recent years, big data and pre-trained large models have shined in the fields of Computer Vision and Natural Language Processing, such as CLIP, DALL·E, and GPT-3, etc., all of which have achieved amazing results, and sequence prediction technology is one of the core modules among them. But for decision intelligence, especially Reinforcement Learning, due to the lack of large datasets similar to CV and NLP and suitable pre-training tasks, the progress of decision-making large models has been slow.

To promote the development of decision-making large models and enhance the practical value of related technologies, many researchers have shifted their focus to the subfield of Offline RL/Batch RL. Offline RL is a reinforcement learning task that trains policies solely through an offline dataset, without any interaction with the environment during the training process. This raises the question: can we learn from some of the research results in the fields of CV and NLP, such as sequence prediction related technologies?

Consequently, in 2021, a series of works represented by Decision Transformer[3]/Trajectory Transformer[1-2] Transformer emerged. These works aimed to simplify decision-making problems into sequence predictions, apply transformer architectures to Reinforcement Learning (RL) tasks, and establish connections with language models such as GPT-x and BERT. Unlike traditional RL, which involves calculating value functions or policy gradients, the DT directly outputs the optimal action selection using a transformer that masks subsequent sequences. By specifying the return-to-go and utilizing state and action information, it can provide the next action and achieve the desired reward. Impressively, DT has not only reached but also surpassed the performance of state-of-the-art (SOTA) model-free offline RL algorithms in environments like Atari and D4RL (MuJoCo).

Quick Facts
-------------
1. DT  is **offline** rl algorithm
2. DT support **discrete** and **continuous** action space
3. The DT utilizes a transformer for action prediction, but it has undergone modifications in the architecture of self-attention.
4. The architecture of the dataset utilized by then DT is dictated by the algorithm’s characteristics. It is imperative that these requirements are met during both the training and testing phases of the model.

Key Equations or Key Graphs
---------------------------
The architecture of DT as following:

.. image:: images/DT.png
   :align: center

The diagram illustrates that when the DT algorithm predicts the action a\ :sub:`t`\ , it is only related to the current timestep’s r\ :sub:`t`\ and s\ :sub:`t`\ , as well as the previous r\ :sub:`t-n`\ , s\ :sub:`t-n`\ , a\ :sub:`t-n`\. It is not related to the subsequent steps. The causal transformer is the module used to achieve this effect.

Pseudo-code
---------------
.. image:: images/DT_algo.png
   :align: center

Implementations
----------------
The default config of DTPolicy as following:

.. autoclass:: ding.policy.dt.DTPolicy
   :noindex:

The neural network interface used is as follows:

.. autoclass:: ding.model.DecisionTransformer
   :members: forward
   :noindex:

Benchmark
-----------
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

P.S.：
The above results were obtained by running the same configuration with three different random seeds(123， 213， 321).

Reference
----------
- Zheng, Q., Zhang, A., & Grover, A. (2022, June). Online decision transformer. In international conference on machine learning (pp. 27042-27059). PMLR.
- https://github.com/kzl/decision-transformer