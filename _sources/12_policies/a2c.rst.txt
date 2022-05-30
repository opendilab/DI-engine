A2C
^^^^^^^

Overview
---------
A3C (Asynchronous advantage actor-critic) algorithm is a simple and lightweight framework for deep reinforcement learning that uses asynchronous gradient descent for optimization of deep neural network controllers. A2C(advantage actor-critic), on the other hand, is the synchronous version of A3C where where the policy gradient algorithm is combined with an advantage function to reduce variance.

Quick Facts
-----------
1. A2C is a **model-free** and **policy-based** RL algorithm.

2. A2C is an **on-policy** algorithm.
   
3. A2C supports both **discrete** and **continuous** action spaces.

4. A2C can be equipped with Recurrent Neural Network (RNN).

Key Equations or Key Graphs
----------------------------
A2C uses advantage estimation in the policy gradient. We implement the advantage by Generalized Advantage Estimation (GAE):

.. math::

   \nabla_{\theta} \log \pi\left(a_{t} \mid {s}_{t} ; \theta\right) \hat{A}^{\pi}\left(s_{t}, {a}_{t} ; \phi \right)


where the k-step advantage function is defined:

.. math::

   \sum_{i=0}^{k-1} \gamma^{i} r_{t+i}+\gamma^{k} \hat{V}^{\pi}\left(s_{t+k} ; \phi\right)-\hat{V}^{\pi}\left(s_{t} ; \phi\right)

Pseudo-code
-----------
.. image:: images/A2C.png
   :align: center
   :height: 150

.. note::
   Different from Q-learning, A2C(and other actor critic methods) alternates between policy estimation and policy improvement.

Extensions
-----------
A2C can be combined with:
    - Multi-step learning
    - RNN
    - Generalized Advantage Estimation (GAE)
      GAE is proposed in `High-Dimensional Continuous Control Using Generalized Advantage Estimation <https://arxiv.org/abs/1506.02438>`_, it uses exponentially-weighted average of different steps of advantage estimators, to make trade-off between variance and bias of the estimation of the advantage:

      .. math::

          \hat{A}_{t}^{\mathrm{GAE}(\gamma, \lambda)}:=(1-\lambda)\left(\hat{A}_{t}^{(1)}+\lambda \hat{A}_{t}^{(2)}+\lambda^{2} \hat{A}_{t}^{(3)}+\ldots\right)

      where the k-step advantage estimator :math:`\hat{A}_t^{(k)}` is defined as :

      .. math::

          \hat{A}_{t}^{(k)}:=\sum_{l=0}^{k-1} \gamma^{l} \delta_{t+l}^{V}=-V\left(s_{t}\right)+r_{t}+\gamma r_{t+1}+\cdots+\gamma^{k-1} r_{t+k-1}+\gamma^{k} V\left(s_{t+k}\right)

      When k=1, the estimator :math:`\hat{A}_t^{(1)}` is the naive advantage estimator:

      .. math::

          \hat{A}_{t}^{(1)}:=\delta_{t}^{V} \quad=-V\left(s_{t}\right)+r_{t}+\gamma V\left(s_{t+1}\right)

      When GAE is used, the common values of :math:`\lambda` usually belong to [0.8, 1.0].


Implementation
------------------
The default config is defined as follows:

    .. autoclass:: ding.policy.a2c.A2CPolicy
        :noindex:

The network interface A2C used is defined as follows:

    .. autoclass:: ding.model.template.vac.VAC
        :members: forward, compute_actor, compute_critic, compute_actor_critic
        :noindex:

The policy gradient and value update of A2C is implemented as follows:

.. code:: python

    def a2c_error(data: namedtuple) -> namedtuple:
        logit, action, value, adv, return_, weight = data
        if weight is None:
            weight = torch.ones_like(value)
        dist = torch.distributions.categorical.Categorical(logits=logit)
        logp = dist.log_prob(action)
        entropy_loss = (dist.entropy() * weight).mean()
        policy_loss = -(logp * adv * weight).mean()
        value_loss = (F.mse_loss(return_, value, reduction='none') * weight).mean()
        return a2c_loss(policy_loss, value_loss, entropy_loss)

.. note::

    we apply GAE to calculate the advantage when update the actor network with the GAE default parameter `gae_lambda` =0.95. 
    The target for the update for the value network is obtained by the value function at the current time step plus the advantage function calculated in collectors. 

Benchmark
-----------


+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| environment         |best mean reward | evaluation results                                  | config link              | comparison           |
+=====================+=================+=====================================================+==========================+======================+
|                     |                 |                                                     |`config_link_p <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Sb3(17)             |
|                     |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|Pong                 |  20             |.. image:: images/benchmark/pong_a2c.png             |atari/config/serial/      |                      |
|                     |                 |                                                     |pong/pong_a2c_config      |                      |
|(PongNoFrameskip-v4) |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_q <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Sb3(3882)           |
|Qbert                |                 |                                                     |DI-engine/tree/main/dizoo/|  Rllib(3620)         |
|                     |  4819           |.. image:: images/benchmark/qbert_a2c.png            |atari/config/serial/      |                      |
|(QbertNoFrameskip-v4)|                 |                                                     |qbert/qbert_a2c_config    |                      |
|                     |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_s <https://  |  Sb3(627)            |
|                     |                 |                                                     |github.com/opendilab/     |  Rllib(692)          |
|SpaceInvaders        |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |  826            |.. image:: images/benchmark/spaceinvaders_a2c.png    |atari/config/serial/      |                      |
|(SpaceInvadersNoFrame|                 |                                                     |spaceinvaders/space       |                      |
|skip-v4)             |                 |                                                     |invaders_a2c_config.py>`_ |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+


P.S.：

1. The above results are obtained by running the same configuration on five different random seeds (0, 1, 2, 3, 4)

References
-----------

Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu: “Asynchronous Methods for Deep Reinforcement Learning”, 2016, ICML 2016; arXiv:1602.01783. https://arxiv.org/abs/1602.01783


Other Public Implementations
----------------------------

- Baselines_
- `sb3`_
- `rllib (Ray)`_
- tianshou_

.. _Baselines: https://github.com/openai/baselines/tree/master/baselines/a2c
.. _sb3: https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/a2c
.. _`rllib (Ray)`: https://github.com/ray-project/ray/blob/master/rllib/agents/a3c/a2c.py
.. _tianshou: https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/a2c.py

