CQL
^^^^^^^

Overview
---------

Offline reinforcement learning (RL) is a re-emerging area of study that aims to learn behaviors using large, previously collected
datasets, without further environment interaction. It has the potential to make tremendous progress in a number of real-world decision-making problems where active data collection is expensive (e.g., in robotics, drug discovery, dialogue generation, recommendation systems) or unsafe/dangerous (e.g., healthcare, autonomous driving, or education).
Besides, the quantities of data that can be gathered
online are substantially lower than the offline datasets. Such a paradigm promises to resolve a key challenge to bringing reinforcement learning algorithms out of constrained lab settings to the real world.

However, directly utilizing existing value-based off-policy RL algorithms in an offline setting generally results
in poor performance, due to issues with bootstrapping from out-of-distribution actions and overfitting. Thus, many constrain techniques are added to basic online RL algorithms. 
Conservative Q-learning (CQL), first proposed in `Conservative Q-Learning for Offline Reinforcement Learning <https://arxiv.org/abs/2006.04779>`_, is one of them which learns conservative Q functions of which the expected value is lower-bounded
via a simple modification to standard value-based RL algorithms.

Quick Facts
-------------
1. CQL is an offline RL algorithm.

2. CQL can be implemented with less than 20 lines of code on top of a
   number of standard, online RL algorithms

3. CQL supports both **discrete** and **continuous** action spaces.


Key Equations or Key Graphs
---------------------------
CQL can be implemented with less than 20 lines of code on top of a
number of standard, online RL algorithms, simply by adding the CQL regularization terms to
the Q-function update.

In general, for the conservative off-policy evaluation, the Q-function is trained via an iterative update:

.. image:: images/cql_policy_evaluation.png
   :align: center
   :scale: 55%

Taking a closer look at the above equation, it consists of two parts - the regularization term and the usual Bellman error with a tradeoff factor alpha. Inside the the regularization term, the first term always pushes the Q value down on the (s,a) pairs sampled from :math:`\mu` whereas the second term pushes Q value up on the (s,a) samples drawn from the offline data set.

According to the following theorem, the above equation lower-bounds the expected value under the policy :math:`\pi`, when :math:`\mu` = :math:`\pi`.

For suitable :math:`\alpha`, the bound holds under sampling
error and function approximation. We also note that as more data becomes available and \|D(s; a)\| increases, the theoretical value of :math:`\alpha` that is needed to guarantee a lower bound decreases, which
indicates that in the limit of infinite data, a lower bound can be obtained by using extremely small
values of :math:`\alpha`.

Note that the analysis presented below assumes that no function approximation is used in the Q-function,
meaning that each iterate can be represented exactly. the result in this theorem can be further generalized to the case of both linear function approximators and non-linear neural network function
approximators, where the latter builds on the neural tangent kernel (NTK) framework. For more details, please refer to the Theorem D.1 and Theorem D.2 in Appendix D.1 in the original paper.


.. image:: images/cql_theorem.png
   :align: center
   :scale: 55%

So, how should we utilize this for policy optimization? We could alternate between performing full off-policy evaluation for each policy iterate, :math:`\hat{\pi}^{k}(a|s)`, and one
step of policy improvement. However, this can be computationally expensive. Alternatively, since the
policy :math:`\hat{\pi}^{k}(a|s)` is typically derived from the Q-function, we could instead choose :math:`\mu(a|s)` to approximate
the policy that would maximize the current Q-function iterate, thus giving rise to an online algorithm. So, for a complete offline RL algorithm, Q-function in general updates as follows:

.. image:: images/cql_general_3.png
   :align: center
   :scale: 55%

where :math:`CQL(R)` is characterized by a particular choice of regularizer :math:`R(\mu)`. If :math:`R(\mu)` is chosen to be the KL-divergence against a prior distribution, :math:`\rho(a|s)`, then we get :math:`\mu(a|s)\approx \rho(a|s)exp(Q(s,a))`. Firstly, if :math:`\rho(a|s)` = Unif(a), then the first term above corresponds to a soft-maximum
of the Q-values at any state s and gives rise to the following variant, called CQL(H):

.. image:: images/cql_equation_4.png
   :align: center
   :scale: 55%

Secondly, if :math:`\rho(a|s)` is chosen to be the previous policy :math:`\hat{\pi}^{k-1}`, the first term in Equation (4) above is replaced by
an exponential weighted average of Q-values of actions from the chosen :math:`\hat{\pi}^{k-1}(a|s)`.

Pseudo-code
---------------
The pseudo-code is shown in Algorithm 1, with differences from conventional actor critic algorithms (e.g., SAC) and deep Q-learning algorithms (e.g.,DQN) in red

.. image:: images/cql.png
   :align: center
   :scale: 55%

The equation (4) in above pseudo-code is:

.. image:: images/cql_equation_4.png
   :align: center
   :scale: 40%

Note that during implementation, the first term in the equation (4) will be computed under `torch.logsumexp`, which consumes lots of running time.

Implementations
----------------
The default config of CQLPolicy is defined as follows:

.. autoclass:: ding.policy.cql.CQLPolicy
   :noindex:


.. autoclass:: ding.policy.cql.CQLDiscretePolicy
   :noindex:


Benchmark
-----------

+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| environment         |best mean reward | evaluation results                                  | config link              | comparison           |
+=====================+=================+=====================================================+==========================+======================+
|                     |                 |                                                     |`config_link_ha <https:// |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|                     |                 |                                                     |DI-engine/blob/main/dizoo/|                      |
|HalfCheetah          |  57.6           |.. image:: images/benchmark/halfcheetah_cql.png      |d4rl/config/halfcheetah_  |   CQL Repo (75.6     |
|                     |  :math:`\pm`    |                                                     |cql_medium_expert         |   :math:`\pm` 25.7)  |
|(Medium Expert)      |  3.7            |                                                     |_config.py>`_             |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_w <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|Walker2d             |                 |                                                     |DI-engine/blob/main/dizoo/|                      |
|                     |  109.7          |.. image:: images/benchmark/walker2d_cql.png         |d4rl/config/walker2d_     |   CQL Repo (107.9    |
|(Medium Expert)      |  :math:`\pm`    |                                                     |cql_medium_expert         |   :math:`\pm` 1.6)   |
|                     |  0.8            |                                                     |_config.py>`_             |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_ho <https:// |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|Hopper               |                 |                                                     |DI-engine/blob/main/dizoo/|                      |
|                     |  85.4           |.. image:: images/benchmark/hopper_cql.png           |d4rl/config/hopper_sac_   |    CQL Repo (105.6   |
|(Medium Expert)      |  :math:`\pm`    |                                                     |cql_medium_expert         |    :math:`\pm` 12.9) |
|                     |  14.8           |                                                     |_config.py>`_             |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+

Specifically for each dataset, our implementation results are as follows:

+---------------------+-----------------+----------------+-----------------+-----------------+----------------------+
| environment         |random           |medium replay   |medium expert    |medium           |expert                |
+=====================+=================+================+=================+=================+======================+
|                     |                 |                |                 |                 |                      |
|HalfCheetah          |18.7 :math:`\pm` |47.1 :math:`\pm`|57.6 :math:`\pm` |49.7 :math:`\pm` |75.1 :math:`\pm`      |
|                     |1.2              |0.3             |3.7              |0.4              |18.4                  |
+---------------------+-----------------+----------------+-----------------+-----------------+----------------------+
|                     |                 |                |                 |                 |                      |
|Walker2d             |22.0 :math:`\pm` |82.6 :math:`\pm`|109.7 :math:`\pm`|82.4 :math:`\pm` |109.2 :math:`\pm`     |
|                     |0.0              |3.4             |0.8              |1.9              |0.3                   |
+---------------------+-----------------+----------------+-----------------+-----------------+----------------------+
|                     |                 |                |                 |                 |                      |
|Hopper               |3.1 :math:`\pm`  |98.3 :math:`\pm`|85.4 :math:`\pm` |79.6 :math:`\pm` |105.4  :math:`\pm`    |
|                     |2.6              |1.8             |14.8             |8.5              |7.2                   |
+---------------------+-----------------+----------------+-----------------+-----------------+----------------------+

P.S.：

1. The above results are obtained by running the same configuration on four different random seeds (5, 10, 20, 30)
2. The above benchmark is for HalfCheetah-v2, Hopper-v2, Walker2d-v2. 
3. The comparison results above is obtained via the paper `Pessimistic Bootstrapping for Uncertainty-Driven Offline Reinforcement Learning  <https://openreview.net/pdf?id=Y4cs1Z3HnqL>`_.
   The complete table is depicted below 

   .. image:: images/cql_official.png
      :align: center
      :scale: 40%  


4. The above Tensorboard results illustrate the unnormalized results 

Reference
----------

- Kumar, Aviral, et al. "Conservative q-learning for offline reinforcement learning." arXiv preprint arXiv:2006.04779 (2020).
- Chenjia Bai, et al. "Pessimistic Bootstrapping for Uncertainty-Driven Offline Reinforcement Learning."


Other Public Implementations
----------------------------

- `CQL release repo`_


.. _`CQL release repo`: https://github.com/aviralkumar2907/CQL