DDPG
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview
---------

Deep Deterministic Policy Gradient (DDPG), proposed in the 2015 paper `Continuous control with deep reinforcement learning <https://arxiv.org/abs/1509.02971>`_, is an algorithm which learns a Q-function and a policy simultaneously.
DDPG is an actor-critic, model-free algorithm based on the deterministic policy gradient(DPG) that can operate over high-dimensional, continuous action spaces.
DPG `Deterministic policy gradient algorithms <http://proceedings.mlr.press/v32/silver14.pdf>`_ algorithm is similar to NFQCA.

Quick Facts
-----------
1. DDPG is only used for environments with **continuous action spaces**.(i.e. MuJoCo)

2. DDPG is an **off-policy** algorithm.

3. DDPG is a **model-free** and **actor-critic** RL algorithm, which optimizes actor network and critic network, respectively.

4. Usually, DDPG use **Ornstein-Uhlenbeck process** or **Gaussian process** (default in our implementation) for exploration.

Key Equations or Key Graphs
---------------------------
The DDPG algorithm maintains a parameterized actor function :math:`\mu\left(s \mid \theta^{\mu}\right)` which specifies the current policy by deterministically mapping states to a specific action. The critic :math:`Q(s, a)` is learned using the Bellman equation as in Q-learning. The actor is updated by following the applying the chain rule to the expected return from the start distribution :math:`J` with respect to the actor parameters:

.. math::
    \begin{aligned}
    \nabla_{\theta^{\mu}} J & \approx \mathbb{E}_{s_{t} \sim \rho^{\beta}}\left[\left.\nabla_{\theta^{\mu}} Q\left(s, a \mid \theta^{Q}\right)\right|_{s=s_{t}, a=\mu\left(s_{t} \mid \theta^{\mu}\right)}\right] \\
    &=\mathbb{E}_{s_{t} \sim \rho^{\beta}}\left[\left.\left.\nabla_{a} Q\left(s, a \mid \theta^{Q}\right)\right|_{s=s_{t}, a=\mu\left(s_{t}\right)} \nabla_{\theta_{\mu}} \mu\left(s \mid \theta^{\mu}\right)\right|_{s=s_{t}}\right]
    \end{aligned}

DDPG uses a **replay buffer** to guarantee that the samples are independently and identically distributed.

To keep neural networks stable in many environments, DDPG uses **“soft” target updates** for **actor-critic** and using. Specifically, DDPG creates a copy of the actor and critic networks, :math:`Q(s, a|\theta^{Q'})` and :math:`\mu' \left(s \mid \theta^{\mu'}\right)` respectively, that are used for calculating the target values. The weights of these target networks are then updated by having them slowly track the learned networks:

.. math::
    \theta' \leftarrow \tau \theta + (1 - \tau)\theta',

where :math:`\tau<<1`. This means that the target values are constrained to change slowly, greatly improving the stability of learning.

A major challenge of learning in continuous action spaces is exploration. The exploration policy is independent from the learning algorithm trough adding noise sampled from a noise process N to actor policy:

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


.. image:: images/DDPG.jpg

Extensions
-----------
DDPG can be combined with:
    - Target Network

        `Continuous control with deep reinforcement learning <https://arxiv.org/abs/1509.02971>`_ proposes soft target updates used to keep the network training stable.
        Since we implement soft update Target Network for actor-critic through ``TargetNetworkWrapper`` in ``model_wrap`` and configuring ``learn.target_theta``.

    - Replay Buffers

        DDPG/TD3 random-collect-size is set to 25000 by default, while it is 25000 for SAC.
        We only simply follow SpinningUp default setting and use random policy to collect initialization data.
        We configure ``random_collect_size`` for data collection.

    - Gaussian noise during collecting transition.

        For the exploration noise process DDPG uses temporally correlated noise in order to explore well in physical environments that have momentum.
        Specifically, DDPG uses Ornstein-Uhlenbeck process with :math:`\theta = 0.15` and :math:`\sigma = 0.2`. The Ornstein-Uhlenbeck process models the velocity of a Brownian particle with friction, which results in temporally correlated values centered around 0.
        However, we use Gaussian noise instead of Ornstein-Uhlenbeck noise due to too many hyper-parameters of Ornstein-Uhlenbeck noise.
        We configure ``collect.noise_sigma`` to control the exploration.


Implementations
----------------
The default config is defined as follows:

.. autoclass:: nervex.policy.ddpg.DDPGPolicy


Model
~~~~~~~~~~~~~~~~~
Here we provide examples of `QAC` model as default model for `DDPG`.

.. autoclass:: nervex.model.qac.q_ac.QAC
    :members: __init__, forward, seed, optimize_actor, compute_q, compute_action, mimic

Train actor-critic model
~~~~~~~~~~~~~~~~~

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
            q_value_dict = {}
            if self._twin_critic:
                q_value_dict['q_value'] = q_value[0].mean()
                q_value_dict['q_value_twin'] = q_value[1].mean()
            else:
                q_value_dict['q_value'] = q_value.mean()
            # target q value. SARSA: first predict next action, then calculate next q value
            with torch.no_grad():
                next_action = self._target_model.forward(next_obs, mode='compute_actor')['action']
                next_data = {'obs': next_obs, 'action': next_action}
                target_q_value = self._target_model.forward(next_data, mode='compute_critic')['q_value']

        - loss computation

        .. code-block:: python

            if self._twin_critic:
                # TD3: two critic networks
                target_q_value = torch.min(target_q_value[0], target_q_value[1])  # find min one as target q value
                # network1
                td_data = v_1step_td_data(q_value[0], target_q_value, reward, data['done'], data['weight'])
                critic_loss, td_error_per_sample1 = v_1step_td_error(td_data, self._gamma)
                loss_dict['critic_loss'] = critic_loss
                # network2(twin network)
                td_data_twin = v_1step_td_data(q_value[1], target_q_value, reward, data['done'], data['weight'])
                critic_twin_loss, td_error_per_sample2 = v_1step_td_error(td_data_twin, self._gamma)
                loss_dict['critic_twin_loss'] = critic_twin_loss
                td_error_per_sample = (td_error_per_sample1 + td_error_per_sample2) / 2
            else:
                # DDPG: single critic network
                td_data = v_1step_td_data(q_value, target_q_value, reward, data['done'], data['weight'])
                critic_loss, td_error_per_sample = v_1step_td_error(td_data, self._gamma)
                loss_dict['critic_loss'] = critic_loss

    2. ``critic network update``

    .. code-block:: python

        self._optimizer_critic.zero_grad()
        for k in loss_dict:
            if 'critic' in k:
                loss_dict[k].backward()
        self._optimizer_critic.step()

    3. ``actor loss``

    .. code-block:: python

        actor_data = self._learn_model.forward(data['obs'], mode='compute_actor')
        actor_data['obs'] = data['obs']
        if self._twin_critic:
            actor_loss = -self._learn_model.forward(actor_data, mode='compute_critic')['q_value'][0].mean()
        else:
            actor_loss = -self._learn_model.forward(actor_data, mode='compute_critic')['q_value'].mean()
        loss_dict['actor_loss'] = actor_loss

    4. ``actor network update``

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

The Benchmark result of DDPG implemented in nerveX is shown in `Benchmark <../feature/algorithm_overview.html>`_

Other Public Implementations
----------------------------

- Baselines_
- rllab_
- `rllib (Ray)`_
- `TD3 release repo`_
- Spinningup_
- tianshou_

.. _Baselines: https://github.com/openai/baselines/tree/master/baselines/ddpg
.. _rllab: https://github.com/rll/rllab/blob/master/rllab/algos/ddpg.py
.. _`rllib (Ray)`: https://github.com/ray-project/ray/tree/master/python/ray/rllib/agents/ddpg
.. _`TD3 release repo`: https://github.com/sfujim/TD3
.. _Spinningup: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ddpg
.. _tianshou: https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/ddpg.py
