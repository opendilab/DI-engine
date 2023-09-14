TD3
^^^^^^^

概述
---------

Twin Delayed DDPG (TD3) 首次在2018年发表的论文 `Addressing Function Approximation Error in Actor-Critic Methods <https://arxiv.org/abs/1802.09477>`_ 中被提出，它是一种考虑了策略和值更新中函数逼近误差之间相互作用的算法。
TD3 是一种基于 `deep deterministic policy gradient (DDPG) <https://arxiv.org/abs/1509.02971>`_ 的 **无模型（model-free）** 算法，属于 **演员—评委（actor-critic）** 类算法中的一员。此算法可以解决高估偏差，时间差分方法中的误差积累以及连续动作空间中对超参数的高敏感性的问题。具体来说，TD3通过引入以下三个关键技巧来解决这些问题:

1. 截断双 Q 学习（Clipped Double-Q Learning）：在计算Bellman误差损失函数中的目标时，TD3 学习两个 Q 函数而不是一个，并使用较小的 Q 值。

2. 延迟的策略更新（Delayed Policy Updates）： TD3更新策略(和目标网络)的频率低于 Q 函数的更新频率。在本文中，作者建议在对 Q 函数更新两次后进行一次策略更新。在我们的实现中，TD3 仅在对 critic 网络更新一定次数 :math:`d` 后，才对策略和目标网络进行一次更新。我们通过配置参数 ``learn.actor_update_freq`` 来实现策略更新延迟。

3. 目标策略平滑（Target Policy Smoothing）：通过沿动作变化平滑 Q 值，TD3 为目标动作引入噪声，使策略更加难以利用 Q 函数的预测错误。

核心要点
-----------
1. TD3 仅支持 **连续动作空间** （例如： MuJoCo）.

2. TD3 是一种 **异策略（off-policy）** 算法.

3. TD3 是一种 **无模型（model-free）** 和 **演员—评委（actor-critic）** 的强化学习算法，它会分别优化策略网络和Q网络。

关键方程或关键框图
---------------------------
TD3 提出了一个截断双 Q 学习变体（Clipped Double-Q Learning），它利用了这样一个概念，即遭受高估偏差的值估计可以用作真实值估计的近似上限。结合下式计算 :math:`Q_{\theta_1}` 的 target，当 :math:`Q_{\theta_2} \textless Q_{\theta_1}` 时，我们认为 :math:`Q_{\theta_1}` 高估了，并将其当作真实值估计的近似上限，取较小的 :math:`Q_{\theta_2}` 计算 :math:`y_1` 以减少过估计。

作为原始版本双 Q 学习的一种拓展，此扩展的动机是，如果目标和当前网络过于相似，例如在actor-critic框架中使用缓慢变化的策略，原始版本的双 Q 学习有时是无效的。

TD3表明，目标网络是深度 Q 学习方法中的一种常见方法，通过减少误差积累来减少目标的方差是至关重要的。

首先，为了解决动作价值估计和策略提升的耦合问题，TD3建议延迟策略更新，直到动作价值估计值尽可能小。因此，TD3只在固定数量次数的 critic 网络更新后再更新策略和目标网络。
我们通过配置参数 ``learn.actor_update_freq`` 来实现策略更新延迟。

其次，截断双 Q 学习（Clipped Double Q-learning）算法的目标更新如下:

.. math::
    y_{1}=r+\gamma \min _{i=1,2} Q_{\theta_{i}^{\prime}}\left(s^{\prime}, \pi_{\phi_{1}}\left(s^{\prime}\right)\right)

在实现中，我们可以通过使用单一的 actor 来优化 :math:`Q_{\theta_1}` 以减少计算开销。由于 TD target 计算过程中使用了同样的策略，因此对于 :math:`Q_{\theta_2}` 的优化目标， :math:`y_2= y_1` 。


最后，确定性策略的一个问题是，由于以神经网络参数化的 Q 函数对 buffer 中动作的价值估计存在突然激增的尖峰（narrow peaks），这会导致策略网络过拟合到这些动作上。并且当更新 critic 网络时，使用确定性策略的学习目标极易受到函数近似误差引起的不准确性的影响，从而增加了目标的方差。
TD3 引入了一种用于深度价值学习的正则化策略，即目标策略平滑，它模仿了SARSA的学习更新。具体来说，TD3通过在目标策略中添加少量随机噪声并在多次计算以下数值后，取平均值来近似此期望：

.. math::
    \begin{array}{l}
    y=r+\gamma Q_{\theta^{\prime}}\left(s^{\prime}, \pi_{\phi^{\prime}}\left(s^{\prime}\right)+\epsilon\right) \\
    \epsilon \sim \operatorname{clip}(\mathcal{N}(0, \sigma),-c, c)
    \end{array}

我们通过配置 ``learn.noise``、 ``learn.noise_sigma`` 和 ``learn.noise_range`` 来实现目标策略平滑。

伪代码
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

扩展
-----------
TD3 可以与以下技术相结合使用:

    - 遵循随机策略的经验回放池初始采集

        在优化模型参数前，我们需要让经验回放池存有足够数目的遵循随机策略的 transition 数据，从而确保在算法初期模型不会对经验回放池数据过拟合。
        DDPG/TD3 的 ``random-collect-size`` 默认设置为25000, SAC 为10000。
        我们只是简单地遵循 SpinningUp 默认设置，并使用随机策略来收集初始化数据。
        我们通过配置 ``random-collect-size`` 来控制初始经验回放池中的 transition 数目。





实现
----------------
默认配置定义如下:

.. autoclass:: ding.policy.td3.TD3Policy
   :noindex:

1. 模型

   在这里，我们提供了 `ContinuousQAC` 模型作为 `TD3` 的默认模型的示例。

    .. autoclass:: ding.model.ContinuousQAC
        :members: forward, compute_actor, compute_critic
        :noindex:

2. 训练 actor-critic 模型

    首先，我们在 ``_init_learn`` 中分别初始化 actor 和 critic 优化器。
    设置两个独立的优化器可以保证我们在计算 actor 损失时只更新 actor 网络参数而不更新 critic 网络，反之亦然。

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

    在 ``_forward_learn`` 中，我们通过计算 critic 损失、更新 critic 网络、计算 actor 损失和更新 actor 网络来更新 actor-critic 策略。
        1. ``critic loss computation``

            - 计算当前值和目标值

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

            - Q 网络目标（**Clipped Double-Q Learning**）和损失计算

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

        3. ``actor loss computation`` 和  ``actor network update`` 取决于策略更新延迟（**delaying the policy updates**）的程度。

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


3. 目标网络（Target Network）

    我们通过 ``_init_learn`` 中的 ``self._target_model`` 初始化来实现目标网络。
    我们配置 ``learn.target_theta`` 来控制平均中的插值因子。


    .. code-block:: python

        # main and target models
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_theta}
        )


4. 目标策略平滑正则（Target Policy Smoothing Regularization）

    我们通过 ``_init_learn`` 中的目标模型初始化来实现目标策略平滑正则。
    我们通过配置 ``learn.noise``、 ``learn.noise_sigma`` 和 ``learn.noise_range`` 来控制引入的噪声，通过对噪声进行截断使所选动作不会太过偏离原始动作。

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



基准
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

1. 上述结果是通过在五个不同的随机种子(0,1,2,3,4)上运行相同的配置获得的。



参考文献
-----------
Scott Fujimoto, Herke van Hoof, David Meger: “Addressing Function Approximation Error in Actor-Critic Methods”, 2018; [http://arxiv.org/abs/1802.09477 arXiv:1802.09477].

其他公开的实现
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
