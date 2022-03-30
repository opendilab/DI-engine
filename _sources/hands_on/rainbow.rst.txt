Rainbow
^^^^^^^

Overview
---------
Rainbow was proposed in `Rainbow: Combining Improvements in Deep Reinforcement Learning <https://arxiv.org/abs/1710.02298>`_. It combines many independent improvements to DQN, including: Double DQN, priority, dueling head, multi-step TD-loss, C51 (distributional RL) and noisy net.

Quick Facts
-----------
1. Rainbow is a **model-free** and **value-based** RL algorithm.

2. Rainbow only support **discrete action spaces**.

3. Rainbow is an **off-policy** algorithm.

4. Usually, Rainbow use **eps-greedy**, **multinomial sample** or **noisy net** for exploration.

5. Rainbow can be equipped with RNN.

6. The DI-engine implementation of Rainbow supports **multi-discrete** action space.

Key Equations or Key Graphs
---------------------------

Double DQN
>>>>>>>>>>>>>>>>>
Double DQN, proposed in `Deep Reinforcement Learning with Double Q-learning <https://arxiv.org/abs/1509.06461>`_, is a common variant of DQN. Conventional DQN maintains a target q network, which is periodically updated with the current q network. Double DQN addresses the overestimation of q-value by decoupling. It selects action with the current q network but estimates the q-value with the target network, formally:

.. math::

   \left(R_{t+1}+\gamma_{t+1} q_{\bar{\theta}}\left(S_{t+1}, \underset{a^{\prime}}{\operatorname{argmax}} q_{\theta}\left(S_{t+1}, a^{\prime}\right)\right)-q_{\theta}\left(S_{t}, A_{t}\right)\right)^{2}


Prioritized Experience Replay(PER)
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

DQN samples uniformly from the replay buffer. Ideally, we want to sample more frequently those transitions from which there is much to learn. As a proxy for learning potential, prioritized experience replay samples transitions with probabilities relative to the last encountered absolute TD error, formally:

.. math::

   p_{t} \propto\left|R_{t+1}+\gamma_{t+1} \max _{a^{\prime}} q_{\bar{\theta}}\left(S_{t+1}, a^{\prime}\right)-q_{\theta}\left(S_{t}, A_{t}\right)\right|^{\omega}


In the original paper of PER, the authors show that PER achieve improvements on most of the 57 Atari games, especially on Gopher, Atlantis, James Bond 007, Space Invaders, etc.

Dueling Network
>>>>>>>>>>>>>>>
The dueling network is a neural network architecture designed for value based RL. It features two streams of computation streams, 
one for state value function :math:`V` and one for the state-dependent action advantage function :math:`A`. 
Both of them share a common convolutional encoder, and are merged by a special aggregator to produce an estimate of the state-action value function :math:`Q` as shown in figure.

.. image:: images/DuelingDQN.png
           :align: center
           :height: 300
           
It is unidentifiable that given :math:`Q` we cannot recover :math:`V` and :math:`A` uniquely. So we force the advantage function zero by the following factorization of action values:

.. math::

   q_{\theta}(s, a)=v_{\eta}\left(f_{\xi}(s)\right)+a_{\psi}\left(f_{\xi}(s), a\right)-\frac{\sum_{a^{\prime}} a_{\psi}\left(f_{\xi}(s), a^{\prime}\right)}{N_{\text {actions }}}

In this way, it can address the issue of identifiability and increase the stability of the optimization.The network architecture of Rainbow is a dueling network architecture adapted for use with return distributions. 

Multi-step Learning
>>>>>>>>>>>>>>>>>>>
A multi-step variant of DQN is then defined by minimizing the alternative loss:


.. math::

   \left(R_{t}^{(n)}+\gamma_{t}^{(n)} \max _{a^{\prime}} q_{\bar{\theta}}\left(S_{t+n}, a^{\prime}\right)-q_{\theta}\left(S_{t}, A_{t}\right)\right)^{2}


where the truncated n-step return is defined as:

.. math::

   R_{t}^{(n)} \equiv \sum_{k=0}^{n-1} \gamma_{t}^{(k)} R_{t+k+1}

In the paper `Revisiting Fundamentals of Experience Replay <https://acsweb.ucsd.edu/~wfedus/pdf/replay.pdf>`_, the authors analyze that a greater capacity of replay buffer substantially increases the performance when multi-step learning is used, and they think the reason is that multi-step learning brings larger variance, which is compensated by a larger replay buffer.

Distribution RL
>>>>>>>>>>>>>>>
Distributional RL was first proposed in `A Distributional Perspective on Reinforcement Learning <https://arxiv.org/abs/1707.06887>`_. It learns to approximate the distribution of returns instead of the expected return using a discrete distribution, whose support is :math:`\boldsymbol{z}`, a vector with :math:`N_{\text {atoms }} \in \mathbb{N}^{+}atoms`, defined by :math:`z^{i}=v_{\min }+(i-1) \frac{v_{\max }-v_{\min }}{N_{\text {atoms }}-1}` for :math:`i \in\left\{1, \ldots, N_{\text {atoms }}\right\}`. The approximate distribution :math:`d_{t}` at time t is defined on this support, with the probability :math:`p_{\theta}^{i}\left(S_{t}, A_{t}\right)` on each atom :math:`i`, such that :math:`d_{t}=\left(z, p_{\theta}\left(S_{t}, A_{t}\right)\right)`. A distributinal variant of Q-learning is then derived by minimizing the Kullbeck-Leibler divergence between the distribution :math:`d_{t}` and the target distribution :math:`d_{t}^{\prime} \equiv\left(R_{t+1}+\gamma_{t+1} z, \quad p_{\bar{\theta}}\left(S_{t+1}, \bar{a}_{t+1}^{*}\right)\right)`, formally:

.. math::

   D_{\mathrm{KL}}\left(\Phi_{\boldsymbol{z}} d_{t}^{\prime} \| d_{t}\right)

Here :math:`\Phi_{\boldsymbol{z}}` is a L2-projection of the target distribution onto the fixed support :math:`\boldsymbol{z}`.

Noisy Net
>>>>>>>>>
Noisy Nets use a noisy linear layer that combines a deterministic and noisy stream:

.. math::

   \boldsymbol{y}=(\boldsymbol{b}+\mathbf{W} \boldsymbol{x})+\left(\boldsymbol{b}_{\text {noisy }} \odot \epsilon^{b}+\left(\mathbf{W}_{\text {noisy }} \odot \epsilon^{w}\right) \boldsymbol{x}\right)

Over time, the network can learn to ignore the noisy stream, but at different rates in different parts of the state space, allowing state-conditional exploration with a form of self-annealing. It usually achieves improvements against :math:`\epsilon`-greedy when the action space is large, e.g. Montezuma's Revenge, because :math:`\epsilon`-greedy tends to quickly converge to a one-hot distribution before the rewards of the large numbers of actions are collected enough.
In our implementation, the noises are resampled before each forward both during data collection and training. When double Q-learning is used, the target network also resamples the noises before each forward. During the noise sampling, the noises are first sampled from :math:`N(0,1)`, then their magnitudes are modulated via a sqrt function with their signs preserved, i.e. :math:`x \rightarrow x.sign() * x.sqrt()`.

Intergrated Method
>>>>>>>>>>>>>>>>>>

First, We replace the 1-step distributional loss with multi-step loss:

.. math::

   \begin{split}
   D_{\mathrm{KL}}\left(\Phi_{\boldsymbol{z}} d_{t}^{(n)} \| d_{t}\right) \\
   d_{t}^{(n)}=\left(R_{t}^{(n)}+\gamma_{t}^{(n)} z,\quad p_{\bar{\theta}}\left(S_{t+n}, a_{t+n}^{*}\right)\right)
   \end{split}

Then, we comine the multi-step distributinal loss with Double DQN by selecting the greedy action using the online network and evaluating such action using the target network.
The KL loss is also used to prioritize the transitions:

.. math::

   p_{t} \propto\left(D_{\mathrm{KL}}\left(\Phi_{z} d_{t}^{(n)} \| d_{t}\right)\right)^{\omega}

The network has a shared representation, which is then fed into a value stream :math:`v_\eta` with :math:`N_{atoms}` outputs, and into an advantage stream :math:`a_{\psi}` with :math:`N_{atoms} \times N_{actions}` outputs, where :math:`a_{\psi}^i(a)` will denote the output corresponding to atom i and action a. For each atom :math:`z_i`, the value and advantage streams are aggregated, as in dueling DQN, and then passed through a softmax layer to obtain the normalized parametric distributions used to estimate the returns’ distributions:

.. math::

  \begin{split}
  p_{\theta}^{i}(s, a)=\frac{\exp \left(v_{\eta}^{i}(\phi)+a_{\psi}^{i}(\phi, a)-\bar{a}_{\psi}^{i}(s)\right)}{\sum_{j} \exp \left(v_{\eta}^{j}(\phi)+a_{\psi}^{j}(\phi, a)-\bar{a}_{\psi}^{j}(s)\right)} \\
  \text { where } \phi=f_{\xi}(s) \text { and } \bar{a}_{\psi}^{i}(s)=\frac{1}{N_{\text {actions }}} \sum_{a^{\prime}} a_{\psi}^{i}\left(\phi, a^{\prime}\right)
  \end{split}


Extensions
-----------
Rainbow can be combined with:
  - RNN

Implementation
---------------
The default config is defined as follows:

.. autoclass:: ding.policy.rainbow.RainbowDQNPolicy
   :noindex:

The network interface Rainbow used is defined as follows:

.. autoclass:: ding.model.template.q_learning.RainbowDQN
   :members: forward
   :noindex:

Benchmark
-----------

+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| environment         |best mean reward | evaluation results                                  | config link              | comparison           |
+=====================+=================+=====================================================+==========================+======================+
|                     |                 |                                                     |`config_link_p <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Tianshou(21)        |
|                     |     21          |                                                     |DI-engine/tree/main/dizoo/|                      |
|Pong                 |                 |.. image:: images/benchmark/pong_rainbow.png         |atari/config/serial/      |                      |
|                     |                 |                                                     |pong/pong_rainbow_config  |                      |
|(PongNoFrameskip-v4) |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_q <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Tianshou(16192.5)   |
|Qbert                |      20600      |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |                 |.. image:: images/benchmark/qbert_rainbow.png        |atari/config/serial/      |                      |
|(QbertNoFrameskip-v4)|                 |                                                     |qbert/qbert_rainbow_config|                      |
|                     |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_s <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Tianshou(1794.5)    |
|SpaceInvaders        |     2168        |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |                 |.. image:: images/benchmark/spaceinvaders_rainbow.png|atari/config/serial/      |                      |
|(SpaceInvadersNoFrame|                 |                                                     |spaceinvaders/spaceinvad  |                      |
|skip-v4)             |                 |                                                     |ers_rainbow_config.py>`_  |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+


P.S.：
    1. The above results are obtained by running the same configuration on five different random seeds (0, 1, 2, 3, 4).
    2. For the discrete action space algorithm, the Atari environment set is generally used for testing (including sub-environments Pong), and Atari environment is generally evaluated by the highest mean reward training 10M ``env_step``. For more details about Atari, please refer to `Atari Env Tutorial <../env_tutorial/atari.html>`_ .

Experiments on Rainbow Tricks
-----------------------------
We conduct experiments on the lunarlander environment using rainbow (dqn) policy to compare the performance of n-step, dueling, priority, and priority_IS tricks with baseline. The code link for the experiments is `here <https://github.com/opendilab/DI-engine/blob/main/dizoo/box2d/lunarlander/config/lunarlander_dqn_config.py>`_.
Note that the config file is set for ``dqn`` by default. If we want to adopt ``rainbow`` policy, we need to change the
type of policy as below.

.. code-block:: python

   lunarlander_dqn_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='rainbow'),
   )


The detailed experiments setting is stated below.

+---------------------+---------------------------------------------------------------------------------------------------+
| Experiments setting | Remark                                                                                            |
+=====================+===================================================================================================+
| base                | one step DQN (n-step=1, dueling=False, priority=False, priority_IS=False)                         |
+---------------------+---------------------------------------------------------------------------------------------------+
| n-step              | n step DQN (n-step=3, dueling=False, priority=False, priority_IS=False)                           |
+---------------------+---------------------------------------------------------------------------------------------------+
| dueling             | use dueling head trick (n-step=3, dueling=True, priority=False, priority_IS=False)                |
+---------------------+---------------------------------------------------------------------------------------------------+
| priority            | use priority experience replay buffer (n-step=3, dueling=False, priority=True, priority_IS=False) |
+---------------------+---------------------------------------------------------------------------------------------------+
| priority_IS         | use importance sampling tricks (n-step=3, dueling=False, priority=True, priority_IS=True)         |
+---------------------+---------------------------------------------------------------------------------------------------+




1. ``reward_mean`` over ``training iteration`` is used as an evaluation metric.

2. Each experiment setting is done for three times with random seed 0, 1, 2 and average the results to ensure stochasticity.

.. code-block:: python

   if __name__ == "__main__":
      serial_pipeline([main_config, create_config], seed=0)

3. By setting the ``exp_name`` in config file, the experiment results can be saved in specified path. Otherwise, it will be saved in ``‘./default_experiment’`` directory.

.. code-block:: python


   from easydict import EasyDict
   from ding.entry import serial_pipeline

   nstep = 1
   lunarlander_dqn_default_config = dict(
    exp_name='lunarlander_exp/base-one-step2',
    env=dict(
       ......



The result is shown in the figure below. As we can see, with tricks on, the speed of convergence is increased by a large amount. In this experiment setting, dueling trick contributes most to the performance.

.. image::
   images/rainbow_exp.png
   :align: center



References
-----------
**(DQN)** Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." 2015; [https://deepmind-data.storage.googleapis.com/assets/papers/DeepMindNature14236Paper.pdf]

**(Rainbow)** Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, David Silver: “Rainbow: Combining Improvements in Deep Reinforcement Learning”, 2017; [http://arxiv.org/abs/1710.02298 arXiv:1710.02298].

**(Double DQN)** Van Hasselt, Hado, Arthur Guez, and David Silver: "Deep reinforcement learning with double q-learning.", 2016; [https://arxiv.org/abs/1509.06461 arXiv:1509.06461]

**(PER)** Schaul, Tom, et al.: "Prioritized Experience Replay.", 2016; [https://arxiv.org/abs/1511.05952 arXiv:1511.05952]

William Fedus, Prajit Ramachandran, Rishabh Agarwal, Yoshua Bengio, Hugo Larochelle, Mark Rowland, Will Dabney: “Revisiting Fundamentals of Experience Replay”, 2020; [http://arxiv.org/abs/2007.06700 arXiv:2007.06700].

**(Dueling network)** Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas: "Dueling network architectures for deep reinforcement learning", 2016; [https://arxiv.org/abs/1511.06581 arXiv:1511.06581]

**(Multi-step)** Sutton, R. S., and Barto, A. G.: "Reinforcement Learning: An Introduction". The MIT press, Cambridge MA. 1998; 

**(Distibutional RL)** Bellemare, Marc G., Will Dabney, and Rémi Munos.: "A distributional perspective on reinforcement learning.", 2017; [https://arxiv.org/abs/1707.06887 arXiv:1707.06887]

**(Noisy net)** Fortunato, Meire, et al.: "Noisy networks for exploration.", 2017; [https://arxiv.org/abs/1706.10295 arXiv:1706.10295]

Other Public Implement
>>>>>>>>>>>>>>>>>>>>>>

- `Tianshou <https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/rainbow.py>`_

- `RLlib <https://github.com/ray-project/ray/blob/master/rllib/agents/dqn/dqn.py>`_









