DDPG
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

概述
---------

DDPG (Deep Deterministic Policy Gradient) 首次在论文
`Continuous control with deep reinforcement learning <https://arxiv.org/abs/1509.02971>`_ 中提出,
是一种同时学习Q函数和策略函数的算法。

DDPG 是基于 DPG (deterministic policy gradient) 的 **无模型（model-free）** 算法，属于 **演员—评委（actor-critic）** 方法中的一员，可以在高维、连续的动作空间上运行。
其中算法 DPG `Deterministic policy gradient algorithms <http://proceedings.mlr.press/v32/silver14.pdf>`_ 与算法 NFQCA `Reinforcement learning in feedback control <https://link.springer.com/content/pdf/10.1007/s10994-011-5235-x.pdf?pdf=button>`_ 相似。

核心要点
-----------
1. DDPG 仅支持 **连续动作空间** （例如： MuJoCo）.

2. DDPG 是一种 **异策略（off-policy）** 算法.

3. DDPG 是一种 **无模型（model-free）** 和 **演员—评委（actor-critic）** 的强化学习算法，它会分别优化策略网络和Q网络。

4. 通常, DDPG 使用 **奥恩斯坦-乌伦贝克过程（Ornstein-Uhlenbeck process）** 或 **高斯过程（Gaussian process）** （在我们的实现中默认使用高斯过程）来探索环境。

关键方程或关键框图
---------------------------
DDPG 包含一个参数化的策略函数（actor） :math:`\mu\left(s \mid \theta^{\mu}\right)` ,
此函数通过将每一个状态确定性地映射到一个具体的动作从而明确当前策略。
此外，算法还包含一个参数化的Q函数（critic） :math:`Q(s, a)` 。
正如 Q-learning 算法，此函数通过贝尔曼方程优化自身。

策略网络通过将链式法则应用于初始分布的预期收益 :math:`J` 来更新自身参数。

具体而言，为了最大化预期收益 :math:`J` ，算法需要计算 :math:`J` 对策略函数参数 :math:`\theta^{\mu}` 的梯度。 :math:`J` 是 :math:`Q(s, a)` 的期望，所以问题转化为计算 :math:`Q^{\mu}(s, \mu(s))` 对 :math:`\theta^{\mu}` 的梯度。

根据链式法则，:math:`\nabla_{\theta^{\mu}} Q^{\mu}(s, \mu(s)) = \nabla_{\theta^{\mu}}\mu(s)\nabla_{a}Q^\mu(s,a)|_{ a=\mu\left(s\right)}+\nabla_{\theta^{\mu}} Q^{\mu}(s, a)|_{ a=\mu\left(s\right)}`。

`Deterministic policy gradient algorithms <http://proceedings.mlr.press/v32/silver14.pdf>`_ 采取了与 `Off-Policy Actor-Critic <https://arxiv.org/pdf/1205.4839.pdf>`_ 中推导 **异策略版本的随机性策略梯度定理** 类似的做法，舍去了上式第二项，
从而得到了近似后的 **确定性策略梯度定理** ：


.. math::
    \begin{aligned}
    \nabla_{\theta^{\mu}} J & \approx \mathbb{E}_{s_{t} \sim \rho^{\beta}}\left[\left.\nabla_{\theta^{\mu}} Q\left(s, a \mid \theta^{Q}\right)\right|_{s=s_{t}, a=\mu\left(s_{t} \mid \theta^{\mu}\right)}\right] \\
    &=\mathbb{E}_{s_{t} \sim \rho^{\beta}}\left[\left.\left.\nabla_{a} Q\left(s, a \mid \theta^{Q}\right)\right|_{s=s_{t}, a=\mu\left(s_{t}\right)} \nabla_{\theta^{\mu}} \mu\left(s \mid \theta^{\mu}\right)\right|_{s=s_{t}}\right]
    \end{aligned}

DDPG 使用了一个 **经验回放池（replay buffer）** 来保证样本分布独立一致。

为了使神经网络稳定优化，DDPG 使用 **软更新（“soft” target updates）** 的方式来优化目标网络，而不是像 DQN 中的 hard target updates 那样定期直接复制网络的参数。
具体而言，DDPG 分别拷贝了 actor 网络 :math:`\mu' \left(s \mid \theta^{\mu'}\right)` 和 critic 网络 :math:`Q'(s, a|\theta^{Q'})` 用于计算目标值。
然后通过让这些目标网络缓慢跟踪学习到的网络来更新这些目标网络的权重：

.. math::
    \theta' \leftarrow \tau \theta + (1 - \tau)\theta',

其中 :math:`\tau<<1`。这意味着目标值被限制为缓慢变化，大大提高了学习的稳定性。

在连续行动空间中学习的一个主要挑战是探索。然而，对于像DDPG这样的 **异策略（off-policy）** 算法来说，它的一个优势是可以独立于算法中的学习过程来处理探索问题。具体来说，我们通过将噪声过程 :math:`\mathcal{N}` 采样的噪声添加到 actor 策略中来构建探索策略:

.. math::
    \mu^{\prime}\left(s_{t}\right)=\mu\left(s_{t} \mid \theta_{t}^{\mu}\right)+\mathcal{N}


伪代码
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

扩展
-----------
DDPG 可以与以下技术相结合使用:
    - 目标网络

        `Continuous control with deep reinforcement learning <https://arxiv.org/abs/1509.02971>`_ 提出了利用软目标更新保持网络训练稳定的方法。
        因此我们通过 ``model_wrap`` 中的 ``TargetNetworkWrapper`` 和配置 ``learn.target_theta`` 来实现 **演员—评委（actor-critic）** 的软更新目标网络。
        
    - 遵循随机策略的经验回放池初始采集

        在优化模型参数前，我们需要让经验回放池存有足够数目的遵循随机策略的 transition 数据，从而确保在算法初期模型不会对经验回放池数据过拟合。
        因此我们通过配置 ``random-collect-size`` 来控制初始经验回放池中的 transition 数目。
        DDPG/TD3 的 ``random-collect-size`` 默认设置为25000, SAC 为10000。
        我们只是简单地遵循 SpinningUp 默认设置，并使用随机策略来收集初始化数据。

    - 采集过渡过程中的高斯噪声

        对于探索噪声过程，DDPG使用时间相关噪声，以提高具有惯性的物理控制问题的探索效率。
        具体而言，DDPG 使用 Ornstein-Uhlenbeck 过程，其中 :math:`\theta = 0.15` 且 :math:`\sigma = 0.2`。Ornstein-Uhlenbeck 过程模拟了带有摩擦的布朗粒子的速度，其结果是以 0 为中心的时间相关值。
        然而，由于 Ornstein-Uhlenbeck 噪声的超参数太多，我们使用高斯噪声代替了 Ornstein-Uhlenbeck 噪声。
        我们通过配置 ``collect.noise_sigma`` 来控制探索程度。


实现
----------------
默认配置定义如下:

.. autoclass:: ding.policy.ddpg.DDPGPolicy
   :noindex:


模型
~~~~~~~~~~~~~~~~~
在这里，我们提供了 `ContinuousQAC` 模型作为 `DDPG` 的默认模型的示例。

.. autoclass:: ding.model.ContinuousQAC
    :members: forward, compute_actor, compute_critic
    :noindex:

训练 actor-critic 模型
~~~~~~~~~~~~~~~~~~~~~~~~~

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
            # target q value. SARSA: first predict next action, then calculate next q value
            with torch.no_grad():
                next_action = self._target_model.forward(next_obs, mode='compute_actor')['action']
                next_data = {'obs': next_obs, 'action': next_action}
                target_q_value = self._target_model.forward(next_data, mode='compute_critic')['q_value']

        - 计算损失

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

目标网络
~~~~~~~~~~~~~~~~~
我们通过 ``_init_learn`` 中的目标模型初始化来实现目标网络。
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

基准
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


P.S.：

1. 上述结果是通过在五个不同的随机种子(0,1,2,3,4)上运行相同的配置获得的。


参考
-----------
Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra: “Continuous control with deep reinforcement learning”, 2015; [http://arxiv.org/abs/1509.02971 arXiv:1509.02971].

David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, et al.. Deterministic Policy Gradient Algorithms. ICML, Jun 2014, Beijing, China. ffhal-00938992f

Hafner, R., Riedmiller, M. Reinforcement learning in feedback control. Mach Learn 84, 137–169 (2011).

Degris, T., White, M., and Sutton, R. S. (2012b). Linear off-policy actor-critic. In 29th International Conference on Machine Learning.

其他公开的实现
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
