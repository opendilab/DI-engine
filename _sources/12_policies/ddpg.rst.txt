DDPG
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview
---------

Deep Deterministic Policy Gradient (DDPG), proposed in the 2015 paper `Continuous control with deep reinforcement learning <https://arxiv.org/abs/1509.02971>`_, is an algorithm which learns a Q-function and a policy simultaneously.
DDPG is an actor-critic, model-free algorithm based on the deterministic policy gradient(DPG) that can operate over high-dimensional, continuous action spaces.
DPG `Deterministic policy gradient algorithms <http://proceedings.mlr.press/v32/silver14.pdf>`_ algorithm is similar to NFQCA `Reinforcement learning in feedback control <https://link.springer.com/content/pdf/10.1007/s10994-011-5235-x.pdf?pdf=button>`_.

Quick Facts
-----------
1. DDPG is only used for environments with **continuous action spaces** (e.g. MuJoCo).

2. DDPG is an **off-policy** algorithm.

3. DDPG is a **model-free** and **actor-critic** RL algorithm, which optimizes the actor network and the critic network, respectively.

4. Usually, DDPG use **Ornstein-Uhlenbeck process** or **Gaussian process** (default in our implementation) for exploration.

Key Equations or Key Graphs
---------------------------
The DDPG algorithm maintains a parameterized actor function :math:`\mu\left(s \mid \theta^{\mu}\right)` which specifies the current policy by deterministically mapping states to a specific action. The critic :math:`Q(s, a)` is learned using the Bellman equation as in Q-learning.

The actor is updated by following the applying the chain rule to the expected return from the start distribution :math:`J` with respect to the actor parameters.

Specifically, to maximize the expected payoff :math:`J`, the algorithm needs to compute the gradient of :math:`J` on the policy function argument :math:`\theta^{\mu}`. :math:`J` is :math:`Q (s, a)` expectations, so the problem is transformed into computing :math:`Q^{\mu} (s, \mu(s))` to :math:`\theta^{\mu}` gradient.

According to the chain rule, :math:`\nabla_{\theta^{\mu}} Q^{\mu}(s,  \mu(s)) = \nabla_{\theta^{\mu}}\mu(s)\nabla_{a}Q^\mu(s,a)|_{ a=\mu\left(s\right)}+\nabla_{\theta^{\mu}} Q^{\mu}(s,  a)|_{ a=\mu\left(s\right)}`.

Similar to the derivation of **off-policy stochastic policy gradient** from `Off-Policy Actor-Critic <https://arxiv.org/pdf/1205.4839.pdf>`_, `Deterministic policy gradient algorithms <http://proceedings.mlr.press/v32/silver14.pdf>`_ dropped the second term.
Thus, the approximate **deterministic policy gradient theorem** is obtained:

.. math::
    \begin{aligned}
    \nabla_{\theta^{\mu}} J & \approx \mathbb{E}_{s_{t} \sim \rho^{\beta}}\left[\left.\nabla_{\theta^{\mu}} Q\left(s, a \mid \theta^{Q}\right)\right|_{s=s_{t}, a=\mu\left(s_{t} \mid \theta^{\mu}\right)}\right] \\
    &=\mathbb{E}_{s_{t} \sim \rho^{\beta}}\left[\left.\left.\nabla_{a} Q\left(s, a \mid \theta^{Q}\right)\right|_{s=s_{t}, a=\mu\left(s_{t}\right)} \nabla_{\theta^{\mu}} \mu\left(s \mid \theta^{\mu}\right)\right|_{s=s_{t}}\right]
    \end{aligned}

DDPG uses a **replay buffer** to guarantee that the samples are independently and identically distributed.

To keep neural networks stable in many environments, DDPG uses **“soft” target updates** to update target networks rather than directly copying the weights. Specifically, DDPG creates a copy of the actor and critic networks, :math:`Q'(s, a|\theta^{Q'})` and :math:`\mu' \left(s \mid \theta^{\mu'}\right)` respectively, that are used for calculating the target values. The weights of these target networks are then updated by having them slowly track the learned networks:

.. math::
    \theta' \leftarrow \tau \theta + (1 - \tau)\theta',

where :math:`\tau<<1`. This means that the target values are constrained to change slowly, greatly improving the stability of learning.

A major challenge of learning in continuous action spaces is exploration. However, it is an advantage for off-policies algorithms such as DDPG that the problem of exploration could be treated independently from the learning algorithm. Specifically, we constructed an exploration policy by adding noise sampled from a noise process N to actor policy:

.. math::
    \mu^{\prime}\left(s_{t}\right)=\mu\left(s_{t} \mid \theta_{t}^{\mu}\right)+\mathcal{N}


Pseudocode
----------

.. math::

    :nowrap:

    \begin{algorithm}[H]
        \caption{Deep Deterministic Policy Gradient}
        \label{alg1}
    \begin{algorithmic}[1]
        \STATE Input: initial policy parameters $\theta$, Q-function parameters $\phi$, empty replay buffer $\mathcal{D}$
        \STATE Set target parameters equal to main parameters $\theta_{\text{targ}} \leftarrow \theta$, $\phi_{\text{targ}} \leftarrow \phi$
        \REPEAT
            \STATE Observe state $s$ and select action $a = \text{clip}(\mu_{\theta}(s) + \epsilon, a_{Low}, a_{High})$, where $\epsilon \sim \mathcal{N}$
            \STATE Execute $a$ in the environment
            \STATE Observe next state $s'$, reward $r$, and done signal $d$ to indicate whether $s'$ is terminal
            \STATE Store $(s,a,r,s',d)$ in replay buffer $\mathcal{D}$
            \STATE If $s'$ is terminal, reset environment state.
            \IF{it's time to update}
                \FOR{however many updates}
                    \STATE Randomly sample a batch of transitions, $B = \{ (s,a,r,s',d) \}$ from $\mathcal{D}$
                    \STATE Compute targets
                    \begin{equation*}
                        y(r,s',d) = r + \gamma (1-d) Q_{\phi_{\text{targ}}}(s', \mu_{\theta_{\text{targ}}}(s'))
                    \end{equation*}
                    \STATE Update Q-function by one step of gradient descent using
                    \begin{equation*}
                        \nabla_{\phi} \frac{1}{|B|}\sum_{(s,a,r,s',d) \in B} \left( Q_{\phi}(s,a) - y(r,s',d) \right)^2
                    \end{equation*}
                    \STATE Update policy by one step of gradient ascent using
                    \begin{equation*}
                        \nabla_{\theta} \frac{1}{|B|}\sum_{s \in B}Q_{\phi}(s, \mu_{\theta}(s))
                    \end{equation*}
                    \STATE Update target networks with
                    \begin{align*}
                        \phi_{\text{targ}} &\leftarrow \rho \phi_{\text{targ}} + (1-\rho) \phi \\
                        \theta_{\text{targ}} &\leftarrow \rho \theta_{\text{targ}} + (1-\rho) \theta
                    \end{align*}
                \ENDFOR
            \ENDIF
        \UNTIL{convergence}
    \end{algorithmic}
    \end{algorithm}


.. image::
   images/DDPG.jpg
   :scale: 75%
   :align: center

Extensions
-----------
DDPG can be combined with:
    - Target Network

        `Continuous control with deep reinforcement learning <https://arxiv.org/abs/1509.02971>`_ proposes soft target updates used to keep the network training stable.
        Since we implement soft update Target Network for actor-critic through ``TargetNetworkWrapper`` in ``model_wrap`` and configuring ``learn.target_theta``.

    - Initial collection of replay buffer following random policy

        Before optimizing the model parameters, we need to have a sufficient number of transition data in the replay buffer following random policy to ensure that the model does not overfit the replay buffer data at the beginning of the algorithm.
        So we control the number of transitions in the initial replay buffer by configuring ``random_collect_size``.
        DDPG/TD3 ``random_collect_size`` is set to 25000 by default, while it is 10000 for SAC.
        We only simply follow SpinningUp default setting and use random policy to collect initialization data.

    - Gaussian noise during collecting transition

        For the exploration noise process DDPG uses temporally correlated noise in order to generate temporally correlated exploration for exploration efficiency in physical control problems with inertia.
        Specifically, DDPG uses Ornstein-Uhlenbeck process with :math:`\theta = 0.15` and :math:`\sigma = 0.2`. The Ornstein-Uhlenbeck process models the velocity of a Brownian particle with friction, which results in temporally correlated values centered around 0.
        However, we use Gaussian noise instead of Ornstein-Uhlenbeck noise due to too many hyper-parameters of Ornstein-Uhlenbeck noise.
        We configure ``collect.noise_sigma`` to control the exploration.


Implementations
----------------
The default config is defined as follows:

.. autoclass:: ding.policy.ddpg.DDPGPolicy
   :noindex:


Model
~~~~~~~~~~~~~~~~~
Here we provide examples of `ContinuousQAC` model as default model for `DDPG`.

.. autoclass:: ding.model.ContinuousQAC
    :members: forward, compute_actor, compute_critic
    :noindex:

Train actor-critic model
~~~~~~~~~~~~~~~~~~~~~~~~~

First, we initialize actor and critic optimizer in ``_init_learn``, respectively.
Setting up two separate optimizers can guarantee that we **only update** actor network parameters and not critic network when we compute actor loss, vice versa.

    .. code-block:: python

        # actor and critic optimizer
        self._optimizer_actor = Adam(
            self._model.actor.parameters(),
            lr=self._cfg.learn.learning_rate_actor,
            weight_decay=self._cfg.learn.weight_decay
        )
        self._optimizer_critic = Adam(
            self._model.critic.parameters(),
            lr=self._cfg.learn.learning_rate_critic,
            weight_decay=self._cfg.learn.weight_decay
        )

In ``_forward_learn`` we update actor-critic policy through computing critic loss, updating critic network, computing actor loss, and updating actor network.
    1. ``critic loss computation``

        - current and target value computation

        .. code-block:: python

            # current q value
            q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']
            # target q value. SARSA: first predict next action, then calculate next q value
            with torch.no_grad():
                next_action = self._target_model.forward(next_obs, mode='compute_actor')['action']
                next_data = {'obs': next_obs, 'action': next_action}
                target_q_value = self._target_model.forward(next_data, mode='compute_critic')['q_value']

        - loss computation

        .. code-block:: python

            # DDPG: single critic network
            td_data = v_1step_td_data(q_value, target_q_value, reward, data['done'], data['weight'])
            critic_loss, td_error_per_sample = v_1step_td_error(td_data, self._gamma)
            loss_dict['critic_loss'] = critic_loss

    2. ``critic network update``

    .. code-block:: python

        self._optimizer_critic.zero_grad()
        loss_dict['critic_loss'].backward()
        self._optimizer_critic.step()

    1. ``actor loss``

    .. code-block:: python

        actor_data = self._learn_model.forward(data['obs'], mode='compute_actor')
        actor_data['obs'] = data['obs']
        actor_loss = -self._learn_model.forward(actor_data, mode='compute_critic')['q_value'].mean()
        loss_dict['actor_loss'] = actor_loss

    1. ``actor network update``

    .. code-block:: python

        # actor update
        self._optimizer_actor.zero_grad()
        actor_loss.backward()
        self._optimizer_actor.step()

Target Network
~~~~~~~~~~~~~~~~~
We implement Target Network trough target model initialization in ``_init_learn``.
We configure ``learn.target_theta`` to control the interpolation factor in averaging.


.. code-block:: python

    # main and target models
    self._target_model = copy.deepcopy(self._model)
    self._target_model = model_wrap(
        self._target_model,
        wrapper_name='target',
        update_type='momentum',
        update_kwargs={'theta': self._cfg.learn.target_theta}
    )

Benchmark
-----------


+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| environment         |best mean reward | evaluation results                                  | config link              | comparison           |
+=====================+=================+=====================================================+==========================+======================+
|                     |                 |                                                     |`config_link_p <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |    Tianshou(11719)   |
|                     |                 |                                                     |DI-engine/tree/main/dizoo/|    Spinning-up(11000)|
|HalfCheetah          |  11334          |.. image:: images/benchmark/halfcheetah_ddpg.png     |mujoco/config/halfcheetah_|                      |
|                     |                 |                                                     |ddpg_default_config.py>`_ |                      |
|(HalfCheetah-v3)     |                 |                                                     |                          |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_q <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |   Tianshou(2197)     |
|Hopper               |                 |                                                     |DI-engine/tree/main/dizoo/|   Spinning-up(1800)  |
|                     |  3516           |.. image:: images/benchmark/hopper_ddpg.png          |mujoco/config/hopper_ddpg_|                      |
|(Hopper-v2)          |                 |                                                     |default_config.py>`_      |                      |
|                     |                 |                                                     |                          |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_s <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |   Tianshou(1401)     |
|Walker2d             |                 |                                                     |DI-engine/tree/main/dizoo/|   Spinning-up(1950)  |
|                     |  3443           |.. image:: images/benchmark/walker2d_ddpg.png        |mujoco/config/walker2d_   |                      |
|(Walker2d-v2)        |                 |                                                     |ddpg_default_config.py>`_ |                      |
|                     |                 |                                                     |                          |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+


P.S.:

1. The above results are obtained by running the same configuration on five different random seeds (0, 1, 2, 3, 4)


References
-----------
Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra: “Continuous control with deep reinforcement learning”, 2015; [http://arxiv.org/abs/1509.02971 arXiv:1509.02971].

David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, et al.. Deterministic Policy Gradient Algorithms. ICML, Jun 2014, Beijing, China. ffhal-00938992f

Hafner, R., Riedmiller, M. Reinforcement learning in feedback control. Mach Learn 84, 137–169 (2011).

Degris, T., White, M., and Sutton, R. S. (2012b). Linear off-policy actor-critic. In 29th International Conference on Machine Learning.

Other Public Implementations
----------------------------

- Baselines_
- `sb3`_
- rllab_
- `rllib (Ray)`_
- `TD3 release repo`_
- Spinningup_
- tianshou_

.. _Baselines: https://github.com/openai/baselines/tree/master/baselines/ddpg
.. _sb3: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ddpg
.. _rllab: https://github.com/rll/rllab/blob/master/rllab/algos/ddpg.py
.. _`rllib (Ray)`: https://github.com/ray-project/ray/tree/master/rllib/agents/ddpg
.. _`TD3 release repo`: https://github.com/sfujim/TD3
.. _Spinningup: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ddpg
.. _tianshou: https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/ddpg.py
