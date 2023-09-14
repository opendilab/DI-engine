TD3
^^^^^^^

Overview
---------

Twin Delayed DDPG (TD3), proposed in the 2018 paper `Addressing Function Approximation Error in Actor-Critic Methods <https://arxiv.org/abs/1802.09477>`_, is an algorithm which considers the interplay between function approximation error in both policy and value updates.
TD3 is an actor-critic, model-free algorithm based on the `deep deterministic policy gradient (DDPG) <https://arxiv.org/abs/1509.02971>`_ that can address overestimation bias, the accumulation of error in temporal difference methods and high sensitivity to hyper-parameters in continuous action spaces. Specifically, TD3 addresses the issue by introducing the following three critical tricks:

1. Clipped Double-Q Learning: When calculating the targets in the Bellman error loss functions, TD3 learns two Q-functions instead of one, and uses the smaller Q-value.

2. Delayed Policy Updates:  TD3 updates the policy (and target networks) less frequently than the Q-function. In the paper, the author recommends one policy update for two Q-function updates. In our implementation, TD3 only updates the policy and target networks after a ﬁxed number of updates :math:`d` to the critic. We implement Policy Updates Delay through configuring ``learn.actor_update_freq``.

3. Target Policy Smoothing:  By smoothing out Q along changes in action, TD3 provides noise to the target action, making it more difficult for the policy to exploit Q-function faults.

Quick Facts
-----------
1. TD3 is only used for environments with **continuous action spaces** (e.g., MuJoCo).

2. TD3 is an **off-policy** algorithm.

3. TD3 is a **model-free** and **actor-critic** RL algorithm, which optimizes actor network and critic network respectively.

Key Equations or Key Graphs
---------------------------
TD3 proposes a clipped Double Q-learning variant which leverages the notion that a value estimate suffering from overestimation bias can be used as an approximate upper-bound to the true value estimate. TD3 shows that target networks, a common approach in deep Q-learning methods, are critical for variance reduction by reducing the accumulation of errors.

Firstly, to address the coupling of value and policy, TD3 proposes delaying policy updates until the value estimate is as small as possible. Therefore, TD3 only updates the policy and target networks after a ﬁxed number of updates :math:`d` to the critic.
We implement Policy Updates Delay through configuring ``learn.actor_update_freq``.

Secondly, the target update of Clipped Double Q-learning algorithm is as follows:

.. math::
    y_{1}=r+\gamma \min _{i=1,2} Q_{\theta_{i}^{\prime}}\left(s^{\prime}, \pi_{\phi_{1}}\left(s^{\prime}\right)\right)

In implementation, computational costs can be reduced by using a single actor optimized with respect to :math:`Q_{\theta_1}` . We then use the same target :math:`y_2= y_1for Q_{\theta_2}`. 


Finally, a concern with deterministic policies is they can overﬁt to narrow peaks in the value estimate. When updating the critic, a learning target using a deterministic policy is highly susceptible to inaccuracies induced by function approximation error, increasing the variance of the target.
TD3 introduces a regularization strategy for deep value learning, target policy smoothing, which mimics the learning update from SARSA. Specifically, TD3 approximates this expectation over actions by adding a small amount of random noise to the target policy and averaging over mini-batches following:

.. math::
    \begin{array}{l}
    y=r+\gamma Q_{\theta^{\prime}}\left(s^{\prime}, \pi_{\phi^{\prime}}\left(s^{\prime}\right)+\epsilon\right) \\
    \epsilon \sim \operatorname{clip}(\mathcal{N}(0, \sigma),-c, c)
    \end{array}

we implement Target Policy Smoothing through configuring ``learn.noise``, ``learn.noise_sigma``, and ``learn.noise_range``.

Pseudocode
----------

.. math::

    :nowrap:

    \begin{algorithm}[H]
        \caption{Twin Delayed DDPG}
        \label{alg1}
    \begin{algorithmic}[1]
        \STATE Input: initial policy parameters $\theta$, Q-function parameters $\phi_1$, $\phi_2$, empty replay buffer $\mathcal{D}$
        \STATE Set target parameters equal to main parameters $\theta_{\text{targ}} \leftarrow \theta$, $\phi_{\text{targ},1} \leftarrow \phi_1$, $\phi_{\text{targ},2} \leftarrow \phi_2$
        \REPEAT
            \STATE Observe state $s$ and select action $a = \text{clip}(\mu_{\theta}(s) + \epsilon, a_{Low}, a_{High})$, where $\epsilon \sim \mathcal{N}$
            \STATE Execute $a$ in the environment
            \STATE Observe next state $s'$, reward $r$, and done signal $d$ to indicate whether $s'$ is terminal
            \STATE Store $(s,a,r,s',d)$ in replay buffer $\mathcal{D}$
            \STATE If $s'$ is terminal, reset environment state.
            \IF{it's time to update}
                \FOR{$j$ in range(however many updates)}
                    \STATE Randomly sample a batch of transitions, $B = \{ (s,a,r,s',d) \}$ from $\mathcal{D}$
                    \STATE Compute target actions
                    \begin{equation*}
                        a'(s') = \text{clip}\left(\mu_{\theta_{\text{targ}}}(s') + \text{clip}(\epsilon,-c,c), a_{Low}, a_{High}\right), \;\;\;\;\; \epsilon \sim \mathcal{N}(0, \sigma)
                    \end{equation*}
                    \STATE Compute targets
                    \begin{equation*}
                        y(r,s',d) = r + \gamma (1-d) \min_{i=1,2} Q_{\phi_{\text{targ},i}}(s', a'(s'))
                    \end{equation*}
                    \STATE Update Q-functions by one step of gradient descent using
                    \begin{align*}
                        & \nabla_{\phi_i} \frac{1}{|B|}\sum_{(s,a,r,s',d) \in B} \left( Q_{\phi_i}(s,a) - y(r,s',d) \right)^2 && \text{for } i=1,2
                    \end{align*}
                    \IF{ $j \mod$ \texttt{policy\_delay} $ = 0$}
                        \STATE Update policy by one step of gradient ascent using
                        \begin{equation*}
                            \nabla_{\theta} \frac{1}{|B|}\sum_{s \in B}Q_{\phi_1}(s, \mu_{\theta}(s))
                        \end{equation*}
                        \STATE Update target networks with
                        \begin{align*}
                            \phi_{\text{targ},i} &\leftarrow \rho \phi_{\text{targ}, i} + (1-\rho) \phi_i && \text{for } i=1,2\\
                            \theta_{\text{targ}} &\leftarrow \rho \theta_{\text{targ}} + (1-\rho) \theta
                        \end{align*}
                    \ENDIF
                \ENDFOR
            \ENDIF
        \UNTIL{convergence}
    \end{algorithmic}
    \end{algorithm}

.. image:: images/TD3.png
   :scale: 80%
   :align: center

Extensions
-----------
TD3 is combined with:

    - Replay Buffers

        DDPG/TD3 ``random_collect_size`` is set to 25000 by default, while it is 10000 for SAC.
        We only simply follow SpinningUp default setting and use random policy to collect initialization data.
        We configure ``random_collect_size`` for data collection.





Implementations
----------------
The default config is defined as follows:

.. autoclass:: ding.policy.td3.TD3Policy

1. Model
   Here we provide examples of `ContinuousQAC` model as default model for `TD3`.

    .. autoclass:: ding.model.ContinuousQAC
        :members: forward, compute_actor, compute_critic
        :noindex:

2. Train actor-critic model

    Firstly, we initialize actor and critic optimizer in ``_init_learn``, respectively.
    Setting up two separate optimizers can guarantee that we **only update** actor network parameters instead of the critic network when we compute actor loss, vice versa.

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

            - target(**Clipped Double-Q Learning**) and loss computation

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

        3. ``actor loss computation`` and  ``actor network update`` depending on the level of **delaying the policy updates**.

            .. code-block:: python

                if (self._forward_learn_cnt + 1) % self._actor_update_freq == 0:
                    actor_data = self._learn_model.forward(data['obs'], mode='compute_actor')
                    actor_data['obs'] = data['obs']
                    if self._twin_critic:
                        actor_loss = -self._learn_model.forward(actor_data, mode='compute_critic')['q_value'][0].mean()
                    else:
                        actor_loss = -self._learn_model.forward(actor_data, mode='compute_critic')['q_value'].mean()

                    loss_dict['actor_loss'] = actor_loss
                    # actor update
                    self._optimizer_actor.zero_grad()
                    actor_loss.backward()
                    self._optimizer_actor.step()


3. Target Network

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


4. Target Policy Smoothing Regularization

    We implement Target Policy Smoothing Regularization trough target model initialization in ``_init_learn``.
    We configure ``learn.noise``, ``learn.noise_sigma``, and ``learn.noise_range`` to control the added noise, which is clipped to keep the target close to the original action.

    .. code-block:: python

        if self._cfg.learn.noise:
            self._target_model = model_wrap(
                self._target_model,
                wrapper_name='action_noise',
                noise_type='gauss',
                noise_kwargs={
                    'mu': 0.0,
                    'sigma': self._cfg.learn.noise_sigma
                },
                noise_range=self._cfg.learn.noise_range
            )



Benchmark
-----------


+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| environment         |best mean reward | evaluation results                                  | config link              | comparison           |
+=====================+=================+=====================================================+==========================+======================+
|                     |                 |                                                     |`config_link_p <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Tianshou(10201)     |
|                     |                 |                                                     |DI-engine/tree/main/dizoo/|  Spinning-up(9750)   |
|HalfCheetah          |  11148          |.. image:: images/benchmark/halfcheetah_td3.png      |mujoco/config/halfcheetah_|  Sb3(9656)           |
|                     |                 |                                                     |td3_default_config.py>`_  |                      |
|(HalfCheetah-v3)     |                 |                                                     |                          |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_q <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |   Tianshou(3472)     |
|Hopper               |                 |                                                     |DI-engine/tree/main/dizoo/|   Spinning-up(3982)  |
|                     |  3720           |.. image:: images/benchmark/hopper_td3.png           |mujoco/config/hopper_td3_ |   sb3(3606 for       |
|(Hopper-v2)          |                 |                                                     |default_config.py>`_      |   Hopper-v3)         |
|                     |                 |                                                     |                          |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_s <https://  |   Tianshou(3982)     |
|                     |                 |                                                     |github.com/opendilab/     |   Spinning-up(3472)  |
|Walker2d             |                 |                                                     |DI-engine/tree/main/dizoo/|   sb3(4718 for       |
|                     |  4386           |.. image:: images/benchmark/walker2d_td3.png         |atari/config/walker2d_td3_|   Walker2d-v2)       |
|(Walker2d-v2)        |                 |                                                     |default_configpy>`_       |                      |
|                     |                 |                                                     |                          |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+


P.S.：

1. The above results are obtained by running the same configuration on five different random seeds (0, 1, 2, 3, 4)



References
-----------
Scott Fujimoto, Herke van Hoof, David Meger: “Addressing Function Approximation Error in Actor-Critic Methods”, 2018; [http://arxiv.org/abs/1802.09477 arXiv:1802.09477].

Other Public Implementations
----------------------------

- `sb3`_
- `rllib (Ray)`_
- `TD3 release repo`_
- Spinningup_
- tianshou_


.. _sb3: https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/td3
.. _`rllib (Ray)`: https://github.com/ray-project/ray/blob/master/rllib/agents/ddpg/td3.py
.. _`TD3 release repo`: https://github.com/sfujim/TD3
.. _Spinningup: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/td3
.. _tianshou: https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/td3.py
