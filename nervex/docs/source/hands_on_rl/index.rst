Hands on RL Algorithm
~~~~~~~~~~~~~~~~~~~~~~~

DQN
^^^^^^^

Overview
---------
DQN was first proposed in 'Playing Atari with Deep Reinforcement Learning <https://arxiv.org/abs/1312.5602>', which combines Q-learning with deep neural network. Different from the previous methods, DQN use a deep neural network to evaluate the q-values, which is updated via TD-loss along with gradient decent.

Quick Facts
-------------
1. DQN is a **model-free** and **value-based** RL algorithm.

2. DQN only support **discrete** action spaces.

3. DQN is an **off-policy** algorithm.

4. Usually, DQN use eps-greedy or multinomial sample for exploration

5. DQN + RNN = DRQN

6. The nerveX implementation of DQN supports multi-discrete action space

Key Equations or Key Graphs
---------------------------
The TD-loss used in DQN is:

.. image:: images/td_loss.png

Pseudo-code
---------------
.. image:: images/DQN.png

.. note::
   Compared with the vanilla version, DQN has been dramatically improved in both algorithm and implementation. In the algorithm part, n-step TD-loss, target network and dueling head are widely used. For the implementation details, the value of epsilon anneals from a high value to zero during the training rather than keeps constant.

Extensions
-----------
- DQN can be combined with:

    * priority replay
    * multi-step TD-loss
    * double(target) Network
    * dueling head
    * RNN

Implementations
----------------
The default config is defined as follows:

.. autoclass:: nervex.policy.dqn.DQNPolicy
