SAC
^^^^^^^

Overview
---------
Soft actor-critic (SAC) is a stable and efficient model-free off-policy maximum entropy actor-critic algorithm for
continuous state and action spaces, which is proposed in the 2018 paper `Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor <https://arxiv.org/abs/1801.01290>`_.
The augmented entropy objective of the policy brings a number of conceptual and practical
advantages including a more powerful exploration and the ability of the policy to capture multiple modes of near optimal
behavior. The authors also showed that this method by combining off-policy
updates with a stable stochastic actor-critic formulation, achieves state-of-the-art performance on a range of continuous control benchmark tasks, outperforming prior on-policy and
off-policy methods.


Quick Facts
-----------
1. SAC is implemented for environments with **continuous** action spaces.(i.e. MuJoCo, Pendulum, and LunarLander)

2. SAC is an **off-policy** and **model-free** algorithm, combined with non-empty replay buffer for policy exploration.

3. SAC is a **actor-critic** RL algorithm, which optimizes actor network and critic network, respectively,

4. SAC is also implemented for **multi-continuous** action space.

Key Equations or Key Graphs
---------------------------
SAC considers a more general maximum entropy objective, which favors stochastic policies by augmenting the objective with the expected entropy of the policy:

.. math::
    J(\pi)=\sum_{t=0}^{T} \mathbb{E}_{\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right) \sim \rho_{\pi}}\left[r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)+\alpha \mathcal{H}\left(\pi\left(\cdot \mid \mathbf{s}_{t}\right)\right)\right].

The temperature parameters :math:`\alpha > 0` controls the stochasticity of the optimal policy.

`Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor <https://arxiv.org/abs/1801.01290>`_ considers a parameterized state value function, soft Q-function, and a tractable policy.
Specifically, the value function and the soft Q-function are modeled as expressive neural networks, and the policy as a Gaussian with mean and covariance given by neural networks.
In particular, SAC applies the reparameterization trick instead of directly minimizing the expected KL-divergence for policy parameters as

.. math::
    J_{\pi}(\phi)=\mathbb{E}_{\mathbf{s}_{t} \sim \mathcal{D}, \epsilon_{t} \sim \mathcal{N}}\left[\log \pi_{\phi}\left(f_{\phi}\left(\epsilon_{t} ; \mathbf{s}_{t}\right) \mid \mathbf{s}_{t}\right)-Q_{\theta}\left(\mathbf{s}_{t}, f_{\phi}\left(\epsilon_{t} ; \mathbf{s}_{t}\right)\right)\right]

We implement reparameterization trick through configuring ``learn.reparameterization``.

.. note::
   Compared with the vanilla version modeling state value function and soft Q-function, our implementation contains two versions. One is modeling state value function and soft Q-function, the other is only modeling soft Q-function through double network.

.. note::

  `Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor <https://arxiv.org/abs/1801.01290>`_ considers a parameterized state value function, soft Q-function, and a tractable policy.
  Our implementation contains two versions. One is modeling state value function and soft Q-function, the other is only modeling soft Q-function through double network.
  We configure ``model.value_network``, ``model.twin_q``, and ``learn.learning_rate_value`` to switch implementation version.


Pseudocode
----------

.. image:: images/SAC-algorithm.png
   :align: center

.. math::

    :nowrap:

    \begin{algorithm}[H]
        \caption{Soft Actor-Critic}
        \label{alg1}
    \begin{algorithmic}[1]
        \STATE Input: initial policy parameters $\theta$, Q-function parameters $\phi_1$, $\phi_2$, empty replay buffer $\mathcal{D}$
        \STATE Set target parameters equal to main parameters $\phi_{\text{targ},1} \leftarrow \phi_1$, $\phi_{\text{targ},2} \leftarrow \phi_2$
        \REPEAT
            \STATE Observe state $s$ and select action $a \sim \pi_{\theta}(\cdot|s)$
            \STATE Execute $a$ in the environment
            \STATE Observe next state $s'$, reward $r$, and done signal $d$ to indicate whether $s'$ is terminal
            \STATE Store $(s,a,r,s',d)$ in replay buffer $\mathcal{D}$
            \STATE If $s'$ is terminal, reset environment state.
            \IF{it's time to update}
                \FOR{$j$ in range(however many updates)}
                    \STATE Randomly sample a batch of transitions, $B = \{ (s,a,r,s',d) \}$ from $\mathcal{D}$
                    \STATE Compute targets for the Q functions:
                    \begin{align*}
                        y (r,s',d) &= r + \gamma (1-d) \left(\min_{i=1,2} Q_{\phi_{\text{targ}, i}} (s', \tilde{a}') - \alpha \log \pi_{\theta}(\tilde{a}'|s')\right), && \tilde{a}' \sim \pi_{\theta}(\cdot|s')
                    \end{align*}
                    \STATE Update Q-functions by one step of gradient descent using
                    \begin{align*}
                        & \nabla_{\phi_i} \frac{1}{|B|}\sum_{(s,a,r,s',d) \in B} \left( Q_{\phi_i}(s,a) - y(r,s',d) \right)^2 && \text{for } i=1,2
                    \end{align*}
                    \STATE Update policy by one step of gradient ascent using
                    \begin{equation*}
                        \nabla_{\theta} \frac{1}{|B|}\sum_{s \in B} \Big(\min_{i=1,2} Q_{\phi_i}(s, \tilde{a}_{\theta}(s)) - \alpha \log \pi_{\theta} \left(\left. \tilde{a}_{\theta}(s) \right| s\right) \Big),
                    \end{equation*}
                    where $\tilde{a}_{\theta}(s)$ is a sample from $\pi_{\theta}(\cdot|s)$ which is differentiable wrt $\theta$ via the reparametrization trick.
                    \STATE Update target networks with
                    \begin{align*}
                        \phi_{\text{targ},i} &\leftarrow \rho \phi_{\text{targ}, i} + (1-\rho) \phi_i && \text{for } i=1,2
                    \end{align*}
                \ENDFOR
            \ENDIF
        \UNTIL{convergence}
    \end{algorithmic}
    \end{algorithm}


.. note::
   Compared with the vanilla version, we only optimize q network and actor network in our second implementation version.


Extensions
-----------
SAC can be combined with:
    - Auto alpha strategy

        `Reinforcement Learning with Deep Energy-Based Policies <https://arxiv.org/abs/1702.08165>`_ proposes entropy coefficient :math:`\alpha` used to determine the relative importance of entropy and reward.
        Extensive experiments conducted by `Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor <https://arxiv.org/abs/1801.01290>`_ demonstrate Soft actor-critic is sensitive to reward scaling since it is related to the temperature of the optimal policy. The optimal reward scale varies between environments, and should be tuned for each task separately.
        Since we implement auto alpha strategy depending on maximum entropy through configuring ``learn.is_auto_alpha`` and ``learn.alpha``.


Implementation
---------------------------------
The default config is defined as follows:

.. autoclass:: ding.policy.sac.SACPolicy

We take the second version implementation(only predict soft Q function) as an example to introduce SAC algorithm:

SAC model includes soft Q network and Policy network:

    Initialization Model.

    .. code-block:: python

        # build network
        self._policy_net = PolicyNet(self._obs_shape, self._act_shape, self._policy_embedding_size)

        self._twin_q = twin_q
        if not self._twin_q:
            self._soft_q_net = SoftQNet(self._obs_shape, self._act_shape, self._soft_q_embedding_size)
        else:
            self._soft_q_net = nn.ModuleList()
            for i in range(2):
                self._soft_q_net.append(SoftQNet(self._obs_shape, self._act_shape, self._soft_q_embedding_size))

        Soft Q prediction from soft Q network:

    .. code-block:: python

        def compute_critic_q(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            action = inputs['action']
            if len(action.shape) == 1:
                action = action.unsqueeze(1)
            state_action_input = torch.cat([inputs['obs'], action], dim=1)
            q_value = self._soft_q_net_forward(state_action_input)
            return {'q_value': q_value}

    Action prediction from policy network:

    .. code-block:: python

        def compute_actor(self, obs: torch.Tensor, deterministic_eval=False, epsilon=1e-6) -> Dict[str, torch.Tensor]:
            mean, log_std = self._policy_net_forward(obs)
            std = log_std.exp()

            # unbounded Gaussian as the action distribution.
            dist = Independent(Normal(mean, std), 1)
            # for reparameterization trick (mean + std * N(0,1))
            if deterministic_eval:
                x = mean
            else:
                x = dist.rsample()
            y = torch.tanh(x)
            action = y

            # epsilon is used to avoid log of zero/negative number.
            y = 1 - y.pow(2) + epsilon
            log_prob = dist.log_prob(x).unsqueeze(-1)
            log_prob = log_prob - torch.log(y).sum(-1, keepdim=True)

            return {'mean': mean, 'log_std': log_std, 'action': action, 'log_prob': log_prob}


    .. note::
        SAC applys an invertible squashing function to the Gaussian samples, and employ the change of variables formula to compute the likelihoods of the bounded actions.
        Specifically, we use unbounded Gaussian as the action distribution through ``Independent(Normal(mean, std), 1)``, which creates a diagonal Normal distribution with the same shape as a Multivariate Normal distribution.
        This is equal to ``log_prob.sum(axis=-1)``.
        Then, the action is squashed by :math:`\tanh(\text{mean})`, and the log-likelihood of action has a simple form :math:`\log \pi(\mathbf{a} \mid \mathbf{s})=\log \mu(\mathbf{u} \mid \mathbf{s})-\sum_{i=1}^{D} \log \left(1-\tanh ^{2}\left(u_{i}\right)\right)`.
        In particular, the ``std`` in SAC is predicted from observation, which is different from PPO(learnable parameter) and TD3(heuristic parameter).


Entropy-Regularized Reinforcement Learning as follows:

    Entropy in target q value.

    .. code-block:: python

        # target q value. SARSA: first predict next action, then calculate next q value
        with torch.no_grad():
            (mu, sigma) = self._learn_model.forward(next_obs, mode='compute_actor')['logit']

            dist = Independent(Normal(mu, sigma), 1)
            pred = dist.rsample()
            next_action = torch.tanh(pred)
            y = 1 - next_action.pow(2) + 1e-6
            # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
            next_log_prob = dist.log_prob(pred).unsqueeze(-1)
            next_log_prob = next_log_prob - torch.log(y).sum(-1, keepdim=True)

            next_data = {'obs': next_obs, 'action': next_action}
            target_q_value = self._target_model.forward(next_data, mode='compute_critic')['q_value']
            # the value of a policy according to the maximum entropy objective
            if self._twin_critic:
                # find min one as target q value
                target_q_value = torch.min(target_q_value[0],
                                           target_q_value[1]) - self._alpha * next_log_prob.squeeze(-1)
            else:
                target_q_value = target_q_value - self._alpha * next_log_prob.squeeze(-1)

    Soft Q value network update.

    .. code-block:: python

        # =================
        # q network
        # =================
        # compute q loss
        if self._twin_q:
            q_data0 = v_1step_td_data(q_value[0], target_value, reward, done, data['weight'])
            loss_dict['q_loss'], td_error_per_sample0 = v_1step_td_error(q_data0, self._gamma)
            q_data1 = v_1step_td_data(q_value[1], target_value, reward, done, data['weight'])
            loss_dict['q_twin_loss'], td_error_per_sample1 = v_1step_td_error(q_data1, self._gamma)
            td_error_per_sample = (td_error_per_sample0 + td_error_per_sample1) / 2
        else:
            q_data = v_1step_td_data(q_value, target_value, reward, done, data['weight'])
            loss_dict['q_loss'], td_error_per_sample = v_1step_td_error(q_data, self._gamma)

        # update q network
        self._optimizer_q.zero_grad()
        loss_dict['q_loss'].backward()
        if self._twin_q:
            loss_dict['q_twin_loss'].backward()
        self._optimizer_q.step()

    Entropy in policy loss.

    .. code-block:: python

        # compute policy loss
        policy_loss = (self._alpha * log_prob - new_q_value.unsqueeze(-1)).mean()

        loss_dict['policy_loss'] = policy_loss

        # update policy network
        self._optimizer_policy.zero_grad()
        loss_dict['policy_loss'].backward()
        self._optimizer_policy.step()

    .. note::
        We implement reparameterization trick trough :math:`(\text{mean} + \text{std} * \mathcal{N}(0,1))`. In particular, the gradient back propagation for ``sigma`` is through ``log_prob`` in policy loss.

Auto alpha strategy

    Alpha initialization through log action shape.

    .. code-block:: python

        if self._cfg.learn.is_auto_alpha:
            self._target_entropy = -np.prod(self._cfg.model.action_shape)
            self._log_alpha = torch.log(torch.tensor([self._cfg.learn.alpha]))
            self._log_alpha = self._log_alpha.to(device='cuda' if self._cuda else 'cpu').requires_grad_()
            self._alpha_optim = torch.optim.Adam([self._log_alpha], lr=self._cfg.learn.learning_rate_alpha)
            self._is_auto_alpha = True
            assert self._log_alpha.shape == torch.Size([1]) and self._log_alpha.requires_grad
            self._alpha = self._log_alpha.detach().exp()

    Alpha update.

    .. code-block:: python

        # compute alpha loss
        if self._is_auto_alpha:
            log_prob = log_prob.detach() + self._target_entropy
            loss_dict['alpha_loss'] = -(self._log_alpha * log_prob).mean()

            self._alpha_optim.zero_grad()
            loss_dict['alpha_loss'].backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()


Benchmark
-----------


+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| environment         |best mean reward | evaluation results                                  | config link              | comparison           |
+=====================+=================+=====================================================+==========================+======================+
|                     |                 |                                                     |`config_link_ha <https:// | Spinning Up (13000)  |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|                     |                 |                                                     |DI-engine/blob/main/dizoo/| SB3(9535)            |
|Halfcheetah          |  12900          |.. image:: images/benchmark/halfcheetah_sac.png      |mujoco/config/halfcheetah_|                      |
|                     |                 |                                                     |sac_default_config.py>`_  | Tianshou(12138)      |
|(Halfcheetah-v3)     |                 |                                                     |                          |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_w <https://  | Spinning Up (5300)   |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|Walker2d             |                 |                                                     |DI-engine/blob/main/dizoo/| SB3(3863)            |
|                     |  5172           |.. image:: images/benchmark/walker2d_sac.png         |mujoco/config/walker2d_   |                      |
|(Walker2d-v2)        |                 |                                                     |sac_default_config.py>`_  | Tianshou(5007)       |
|                     |                 |                                                     |                          |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_ho <https:// | Spinning Up (3500)   |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|Hopper               |                 |                                                     |DI-engine/blob/main/dizoo/| SB3(2325)            |
|                     |  3653           |.. image:: images/benchmark/hopper_sac.png           |mujoco/config/hopper_sac_ |                      |
|(Hopper-v2)          |                 |                                                     |default_config.py>`_      | Tianshou(3542)       |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+

Reference
---------
- Haarnoja, et al. Soft Actor-Critic Algorithms and Applications. [https://arxiv.org/abs/1812.05905 arXiv:1812.05905], 2019.

- Haarnoja, et al. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. [https://arxiv.org/abs/1801.01290 arXiv:1801.01290], 2018.

Other Public Implementations
----------------------------

- Baselines_
- rllab_
- `rllib (Ray)`_
- Spinningup_
- tianshou_

.. _Baselines: https://github.com/openai/baselines/tree/master/baselines/sac
.. _rllab: https://github.com/rll/rllab/blob/master/rllab/algos/sac.py
.. _`rllib (Ray)`: https://github.com/ray-project/ray/tree/master/python/ray/rllib/agents/sac
.. _Spinningup: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac
.. _tianshou: https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/sac.py
