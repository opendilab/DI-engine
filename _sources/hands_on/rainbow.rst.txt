Rainbow
^^^^^^^

Overview
---------
Rainbow was proposed in `Rainbow: Combining Improvements in Deep Reinforcement Learning <https://arxiv.org/abs/1710.02298>`_. It combines many independent improvements to DQN, including: target network(double DQN), priority, dueling head, multi-step TD-loss, C51 and noisy net.

Quick Facts
-----------
1. Rainbow is a **model-free** and **value-based** RL algorithm.

2. Rainbow only support **discrete action spaces**.

3. Rainbow is an **off-policy** algorithm.

4. Usually, Rainbow use **eps-greedy**, **multinomial sample** or **noisy net** for exploration.

5. Rainbow can be equipped with RNN.

6. The DI-engine implementation of Rainbow supports **multi-discrete** action space.

Double Q-learning
------------------
Double Q-learning maintains a target q network, which is periodically updated with the current q network. Double Q-learning decouples the over-estimation of q-value by selects action with the current q network but estimate the q-value with the target network, formally:

.. math::

   \left(R_{t+1}+\gamma_{t+1} q_{\bar{\theta}}\left(S_{t+1}, \underset{a^{\prime}}{\operatorname{argmax}} q_{\theta}\left(S_{t+1}, a^{\prime}\right)\right)-q_{0}\left(S_{t}, A_{t}\right)\right)^{2}

Prioritized Experience Replay(PER)
----------------------------------
DQN samples uniformly from the replay buffer. Ideally, we want to sample more frequently those transitions from which there is much to learn. As a proxy for learning potential, prioritized experience replay samples transitions with probability relative to the last encountered absolute TD error, formally:

.. math::
   
   p_{t} \propto\left|R_{t+1}+\gamma_{t+1} \max _{a^{\prime}} q_{\bar{\theta}}\left(S_{t+1}, a^{\prime}\right)-q_{\theta}\left(S_{t}, A_{t}\right)\right|^{\omega}
   

The original paper of PER, the authors show that PER achieve improvements on most of the 57 Atari games, especially on Gopher, Atlantis, James Bond 007, Space Invaders, etc.

Dueling Network
---------------
The dueling network is a neural network architecture designed for value based RL. It features two streams of computation, the value and advantage
streams, sharing a convolutional encoder, and merged by a special aggregator. This corresponds to the following factorization of action values:

.. math::

   q_{\theta}(s, a)=v_{\eta}\left(f_{\xi}(s)\right)+a_{\psi}\left(f_{\xi}(s), a\right)-\frac{\sum_{a^{\prime}} a_{\psi}\left(f_{\xi}(s), a^{\prime}\right)}{N_{\text {actions }}}
   
The network architecture of Rainbow is a dueling network architecture adapted for use with return distributions. The network has a shared representation, which is then fed into a value stream :math:`v_\eta` with :math:`N_{atoms}` outputs, and into an advantage stream :math:`a_{\psi}` with :math:`N_{atoms} \times N_{actions}` outputs, where :math:`a_{\psi}^i(a)` will denote the output corresponding to atom i and action a. For each atom :math:`z_i`, the value and advantage streams are aggregated, as in dueling DQN, and then passed through a softmax layer to obtain the normalized parametric distributions used to estimate the returns’ distributions:

.. math::

  \begin{array}{r}
  p_{\theta}^{i}(s, a)=\frac{\exp \left(v_{\eta}^{i}(\phi)+a_{\psi}^{i}(\phi, a)-\bar{a}_{\psi}^{i}(s)\right)}{\sum_{j} \exp \left(v_{\eta}^{j}(\phi)+a_{\psi}^{j}(\phi, a)-\bar{a}_{\psi}^{j}(s)\right)} \\
  \text { where } \phi=f_{\xi}(s) \text { and } \bar{a}_{\psi}^{i}(s)=\frac{1}{N_{\text {actions }}} \sum_{a^{\prime}} a_{\psi}^{i}\left(\phi, a^{\prime}\right)
  \end{array}

Multi-step Learning
-------------------
A multi-step variant of DQN is then defined by minimizing the alternative loss:

   
.. math::

   \left(R_{t}^{(n)}+\gamma_{t}^{(n)} \max _{a^{\prime}} q_{\bar{\theta}}\left(S_{t+n}, a^{\prime}\right)-q_{\theta}\left(S_{t}, A_{t}\right)\right)^{2}


where the truncated n-step return is defined as:

.. math::

   `R_{t}^{(n)} \equiv \sum^{n-1} \gamma_{t}^{(k)} R_{t+k+1}

In the paper `Revisiting Fundamentals of Experience Replay <https://acsweb.ucsd.edu/~wfedus/pdf/replay.pdf>`_, the authors analyze that a greater capacity of replay buffer substantially increase the performance when multi-step learning is used, and they think the reason is that multi-step learning brings larger variance, which is compensated by a larger replay buffer.

Noisy Net
---------
Noisy Nets use a noisy linear layer that combines a deterministic and noisy stream:

.. math::
   
   \boldsymbol{y}=(\boldsymbol{b}+\mathbf{W} \boldsymbol{x})+\left(\boldsymbol{b}_{\text {noisy }} \odot \epsilon^{b}+\left(\mathbf{W}_{\text {noisy }} \odot \epsilon^{w}\right) \boldsymbol{x}\right)

Over time, the network can learn to ignore the noisy stream, but at different rates in different parts of the state space, allowing state-conditional exploration with a form of self-annealing. It usually achieves improvements against epsilon-greedy when the action space is large, e.g. Montezuma's Revenge, because epsilon-greedy tends to quickly converge to a one-hot distribution before the rewards of the large numbers of actions are collected enough.
In our implementation, the noises are resampled before each forward both during data collection and training. When double Q-learning is used, the target network also resamples the noises before each forward. During the noise sampling, the nosies are first sampled form N(0,1), then their magnitudes are modulated via a sqrt function with their signs preserved, i.e. x -> x.sign() * x.sqrt().

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
   :members: __init__, forward
   :noindex:

The Benchmark result of Rainbow implemented in DI-engine is shown in `Benchmark <../feature/algorithm_overview_en.html>`_


References
-----------
Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, David Silver: “Rainbow: Combining Improvements in Deep Reinforcement Learning”, 2017; [http://arxiv.org/abs/1710.02298 arXiv:1710.02298].

William Fedus, Prajit Ramachandran, Rishabh Agarwal, Yoshua Bengio, Hugo Larochelle, Mark Rowland, Will Dabney: “Revisiting Fundamentals of Experience Replay”, 2020; [http://arxiv.org/abs/2007.06700 arXiv:2007.06700].
