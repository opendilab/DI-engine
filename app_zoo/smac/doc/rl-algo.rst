RL Algorithm
~~~~~~~~~~~~

DQN
^^^^^^^

Overview
---------
DQN was first proposed in 'Playing Atari with Deep Reinforcement Learning <https://arxiv.org/abs/1312.5602>', which combines Q-learning with deep neural network. Different from the previous methods, DQN use a deep neural network to evaluate the q-values, which is updated via TD-loss along with gradient decent.

Quick Facts
-----------
DQN is a model-free and value-based RL algorithm.

DQN only support discrete action spaces.

Pseudo-code
-----------
.. image:: images/DQN.png

.. note::
   Compared with the vanilla version, DQN has been dramatically improved in both algorithm and implementation. In the algorithm part, n-step TD-loss, target network and dueling head are widely used. For the implementation details, the value of epsilon anneals from a high value to zero during the training rather than keeps constant.

Extensions
-----------
- DQN can be combined with:
    * priority replay
    * multi-step TD-loss
    * double(target) Network
    * dueling head

Implementation
------------
  The default config is defined as follows:

  .. code:: python

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='dqn',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether priority replay is used.
        priority=False,
        # (float) Reward's future discount factor, aka. gamma.
        discount_factor=0.97,
        # (int) N-step reward for target q_value estimation
        nstep=1,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after each collection of the collector .
            # collect data -> update policy-> collect data -> ...
            update_per_collect=3,
            # The batch size for each training iteration.
            batch_size=64,
            learning_rate=0.001,
            # (float) The weight of L2 reg loss of the network parameters.
            weight_decay=0.0,
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
            # (int) Frequence of target network update.
            target_update_freq=100,
            # (bool) Whether ignore done(usually for max step termination env)
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) Only one of [n_sample, n_step, n_episode] shoule be set
            n_sample=8,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (bool) Whether to use hindsight experience replay
            her=False,
            her_strategy='future',
            her_replay_k=1,
        ),
        eval=dict(),
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # (str) Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                # (int) Decay steps(env step).
                decay=10000,
            ),
            replay_buffer=dict(
                # (int) Replay buffer size.
                replay_buffer_size=10000,
            )
        ),
    )
















C51
^^^^^^^

Overview
---------
C51 was first proposed in 'A Distributional Perspective on Reinforcement Learning <https://arxiv.org/abs/1707.06887>', different from previous works, C51 evaluates the complete distribution of a q-value rather than only the expectation.

Quick Facts
-----------
C51 is a model-free and value-based RL algorithm.

C51 only support discrete action spaces.

Pseudo-code
---------
.. image:: images/C51.png

.. note::
   C51 models the value distribution using a discrete distribution, whose support set are N atoms: z_i = V_min + i * delta, i = 0,1,...,N-1 and delta = (V_max - V_min) / N. Each atom z_i has a parameterized probability p_i. The Bellman update of C51 projects the distribution of r + gamma * z_j^(t+1) onto the distribution z_i^t.

Extensions
-----------
- C51s can be combined with:
   * priority replay
   * multi-step TD-loss
   * double(target) Network
   * dueling head

Implementation
------------
  The default config is defined as follows:

  .. code:: python

      config = dict(
          # (str) RL policy register name (refer to function "POLICY_REGISTRY").
          type='c51',
          # (bool) Whether to use cuda for network.
          cuda=False,
          # (bool) Whether the RL algorithm is on-policy or off-policy.
          on_policy=False,
          # (bool) Whether priority replay is used.
          priority=False,
          # (float) Reward's future discount factor, aka. gamma.
          model=dict(
              # (float) Value of the smallest atom in the support set.
              # Default to -10.0.
              v_min=-10,
              # (float) Value of the biggest atom in the support set.
              # Default to 10.0.
              v_max=10,
              # (int) Number of atoms in the support set of the
              # value distribution. Default to 51.
              n_atom=51,
          ),
          discount_factor=0.97,
          # (int) N-step reward for target q_value estimation
          nstep=1,
          learn=dict(
              # (bool) Whether to use multi gpu
              multi_gpu=False,
              # How many updates(iterations) to train after each collection of the collector .
              # collect data -> update policy-> collect data -> ...
              update_per_collect=3,
              # The batch size for each training iteration.
              batch_size=64,
              learning_rate=0.001,
              # (float) The weight of L2 reg loss of the network parameters.
              weight_decay=0.0,
              # ==============================================================
              # The following configs are algorithm-specific
              # ==============================================================
              # (int) Frequence of target network update.
              target_update_freq=100,
              # (bool) Whether ignore done(usually for max step termination env)
              ignore_done=False,
          ),
          # collect_mode config
          collect=dict(
              # (int) Only one of [n_sample, n_step, n_episode] shoule be set
              n_sample=8,
              # (int) Cut trajectories into pieces with length "unroll_len".
              unroll_len=1,
              # ==============================================================
              # The following configs is algorithm-specific
              # ==============================================================
              # (bool) Whether to use hindsight experience replay
              her=False,
              her_strategy='future',
              her_replay_k=1,
          ),
          eval=dict(),
          # other config
          other=dict(
              # Epsilon greedy with decay.
              eps=dict(
                  # (str) Decay type. Support ['exp', 'linear'].
                  type='exp',
                  start=0.95,
                  end=0.1,
                  # (int) Decay steps(env step).
                  decay=10000,
              ),
              replay_buffer=dict(
                  # (int) Replay buffer size.
                  replay_buffer_size=10000,
              )
          ),
      )

  The bellman updates of C51 is implemented as:

  .. code:: python

    def dist_nstep_td_error(
            data: namedtuple,
            gamma: float,
            v_min: float,
            v_max: float,
            n_atom: int,
            nstep: int = 1,
            value_gamma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Overview:
            Multistep (1 step or n step) td_error for distributed q-learning based algorithm
        Arguments:
            - data (:obj:`dist_nstep_td_data`): the input data, dist_nstep_td_data to calculate loss
            - gamma (:obj:`float`): discount factor
            - nstep (:obj:`int`): nstep num, default set to 1
        Returns:
            - loss (:obj:`torch.Tensor`): nstep td error, 0-dim tensor
        Shapes:
            - data (:obj:`dist_nstep_td_data`): the dist_nstep_td_data containing\
                ['dist', 'next_n_dist', 'act', 'reward', 'done', 'weight']
            - dist (:obj:`torch.FloatTensor`): :math:`(B, N, n_atom)` i.e. [batch_size, action_dim, n_atom]
            - next_n_dist (:obj:`torch.FloatTensor`): :math:`(B, N, n_atom)`
            - act (:obj:`torch.LongTensor`): :math:`(B, )`
            - next_n_act (:obj:`torch.LongTensor`): :math:`(B, )`
            - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is timestep(nstep)
            - done (:obj:`torch.BoolTensor`) :math:`(B, )`, whether done in last timestep
        """
        dist, next_n_dist, act, next_n_act, reward, done, weight = data
        device = reward.device
        assert len(act.shape) == 1, act.shape
        reward_factor = torch.ones(nstep).to(device)
        for i in range(1, nstep):
            reward_factor[i] = gamma * reward_factor[i - 1]
        reward = torch.matmul(reward_factor, reward)
        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)
        support = torch.linspace(v_min, v_max, n_atom).to(device)
        delta_z = (v_max - v_min) / (n_atom - 1)
        batch_size = act.shape[0]
        batch_range = torch.arange(batch_size)
        if weight is None:
            weight = torch.ones_like(reward)

        next_n_dist = next_n_dist[batch_range, next_n_act].detach()

        if value_gamma is None:
            target_z = reward + (1 - done) * (gamma ** nstep) * support
        else:
            value_gamma = value_gamma.unsqueeze(-1)
            target_z = reward + (1 - done) * value_gamma * support
        target_z = target_z.clamp(min=v_min, max=v_max)
        b = (target_z - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()
        # Fix disappearing probability mass when l = b = u (b is int)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (n_atom - 1)) * (l == u)] += 1

        proj_dist = torch.zeros_like(next_n_dist)
        offset = torch.linspace(0, (batch_size - 1) * n_atom, batch_size).unsqueeze(1).expand(batch_size,
                                                                                              n_atom).long().to(device)
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_n_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_n_dist * (b - l.float())).view(-1))

        assert (dist[batch_range, act] > 0.0).all(), ("dist act", dist[batch_range, act], "dist:", dist)
        log_p = torch.log(dist[batch_range, act])

        if len(weight.shape) == 1:
            weight = weight.unsqueeze(-1)

        td_error_per_sample = -(log_p * proj_dist).sum(-1)

        loss = -(log_p * proj_dist * weight).sum(-1).mean()

        return loss, td_error_per_sample

QRDQN
^^^^^^^

Overview
---------
QR(Quantile Regression)DQN was proposed in 'Distributional Reinforcement Learning with Quantile Regression <https://arxiv.org/pdf/1710.10044>' and inherits the idea of learning the distribution of a q-value. Instead of approximate the distribution density function with discrete atoms, QRDQN, direct regresses a discrete set of quantiles of a q-value.

Quick Facts
-----------
QRDQN is a model-free and value-based RL algorithm.

QRDQN only support discrete action spaces.

Key Equations
-------------
The quantile regression loss, for a quantile tau in [0, 1], is an asymmetric convex loss function that penalizes overestimation errors with weight tau and underestimation errors with weight 1−tau. For a distribution Z, and a given quantile tau, the value of the quantile function F_Z^−1(tau) may be characterized
as the minimizer of the quantile regression loss:
.. image:: images/QR.png

Pseudo-code
---------
.. image:: images/QRDQN.png

.. note::
   The quantile huber loss is applied during the Bellman update of QRDQN.

Extensions
-----------
- QRDQN can be combined with:
  * priority replay
  * multi-step TD-loss
  * double(target) Network

Implementation
------------
  The default config is defined as follows:

  .. code:: python

      config = dict(
          # (str) RL policy register name (refer to function "POLICY_REGISTRY").
          type='qrdqn',
          # (bool) Whether to use cuda for network.
          cuda=False,
          # (bool) Whether the RL algorithm is on-policy or off-policy.
          on_policy=False,
          # (bool) Whether priority replay is used.
          priority=False,
          # (float) Reward's future discount factor, aka. gamma.
          model=dict(
              # (int) Number of quantiles of the
              # value distribution. Default to 32.
              num_quantiles=32,
          ),
          discount_factor=0.97,
          # (int) N-step reward for target q_value estimation
          nstep=1,
          learn=dict(
              # (bool) Whether to use multi gpu
              multi_gpu=False,
              # How many updates(iterations) to train after each collection of the collector .
              # collect data -> update policy-> collect data -> ...
              update_per_collect=3,
              # The batch size for each training iteration.
              batch_size=64,
              learning_rate=0.001,
              # (float) The weight of L2 reg loss of the network parameters.
              weight_decay=0.0,
              # ==============================================================
              # The following configs are algorithm-specific
              # ==============================================================
              # (int) Frequence of target network update.
              target_update_freq=100,
              # (bool) Whether ignore done(usually for max step termination env)
              ignore_done=False,
          ),
          # collect_mode config
          collect=dict(
              # (int) Only one of [n_sample, n_step, n_episode] shoule be set
              n_sample=8,
              # (int) Cut trajectories into pieces with length "unroll_len".
              unroll_len=1,
              # ==============================================================
              # The following configs is algorithm-specific
              # ==============================================================
              # (bool) Whether to use hindsight experience replay
              her=False,
              her_strategy='future',
              her_replay_k=1,
          ),
          eval=dict(),
          # other config
          other=dict(
              # Epsilon greedy with decay.
              eps=dict(
                  # (str) Decay type. Support ['exp', 'linear'].
                  type='exp',
                  start=0.95,
                  end=0.1,
                  # (int) Decay steps(env step).
                  decay=10000,
              ),
              replay_buffer=dict(
                  # (int) Replay buffer size.
                  replay_buffer_size=10000,
              )
          ),
      )

  The bellman updates of QRDQN is implemented as:

  .. code:: python

    def qrdqn_nstep_td_error(
            data: namedtuple,
            gamma: float,
            nstep: int = 1,
    ) -> torch.Tensor:
        """
        Overview:
            Multistep (1 step or n step) td_error with in QRDQN
        Arguments:
            - data (:obj:`iqn_nstep_td_data`): the input data, iqn_nstep_td_data to calculate loss
            - gamma (:obj:`float`): discount factor
            - nstep (:obj:`int`): nstep num, default set to 1
        Returns:
            - loss (:obj:`torch.Tensor`): nstep td error, 0-dim tensor
        Shapes:
            - data (:obj:`q_nstep_td_data`): the q_nstep_td_data containing\
            ['q', 'next_n_q', 'action', 'reward', 'done']
            - q (:obj:`torch.FloatTensor`): :math:`(tau, B, N)` i.e. [tau x batch_size, action_dim]
            - next_n_q (:obj:`torch.FloatTensor`): :math:`(tau', B, N)`
            - action (:obj:`torch.LongTensor`): :math:`(B, )`
            - next_n_action (:obj:`torch.LongTensor`): :math:`(B, )`
            - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is timestep(nstep)
            - done (:obj:`torch.BoolTensor`) :math:`(B, )`, whether done in last timestep
        """
        q, next_n_q, action, next_n_action, reward, done, tau, weight = data

        assert len(action.shape) == 1, action.shape
        assert len(next_n_action.shape) == 1, next_n_action.shape
        assert len(done.shape) == 1, done.shape
        assert len(q.shape) == 3, q.shape
        assert len(next_n_q.shape) == 3, next_n_q.shape
        assert len(reward.shape) == 2, reward.shape

        if weight is None:
            weight = torch.ones_like(action)

        batch_range = torch.arange(action.shape[0])

        # shape: batch_size x num x 1
        q_s_a = q[batch_range, action, :].unsqueeze(2)
        # shape: batch_size x 1 x num
        target_q_s_a = next_n_q[batch_range, next_n_action, :].unsqueeze(1)

        assert reward.shape[0] == nstep
        reward_factor = torch.ones(nstep).to(reward)
        for i in range(1, nstep):
            reward_factor[i] = gamma * reward_factor[i - 1]
        # shape: batch_size
        reward = torch.matmul(reward_factor, reward)
        # shape: batch_size x 1 x num
        target_q_s_a = reward.unsqueeze(-1).unsqueeze(-1) + (gamma ** nstep) * target_q_s_a * (1 - done).unsqueeze(-1).unsqueeze(-1)

        # shape: batch_size x num x num
        u = F.smooth_l1_loss(target_q_s_a, q_s_a, reduction="none")
        # shape: batch_size
        loss = (u * (
                tau - (target_q_s_a - q_s_a).detach().le(0.).float()
            ).abs()).sum(-1).mean(1)

        return (loss * weight).mean(), loss

IQN
^^^^^^^

Overview
---------
IQN was proposed in 'Implicit Quantile Networks for Distributional Reinforcement Learning <https://arxiv.org/pdf/1806.06923>'. The key difference between IQN and QRDQN is that IQN introduces the implicit quantile network (IQN), a deterministic parametric function trained to re-parameterize samples from a base distribution, e.g. tau in U([0, 1]), to the respective quantile values of a target distribution, while QRDQN direct learns a fixed set of pre-defined quantiles.

Quick Facts
-----------
IQN is a model-free and value-based RL algorithm.

IQN only support discrete action spaces.

Key Equations
-------------
In implicit quantile networks, a sampled quantile tau is first encoded into an embedding vector via:
.. image:: images/IQN.png
Then the quantile embedding is element-wise multiplied by the embedding of the observation of the environment, and the subsequent fully-connected layers map the resulted product vector to the respective quantile value.


Extensions
-----------
- IQN can be combined with:
  * priority replay
  * multi-step TD-loss
  * double(target) Network

Implementation
------------
  The default config is defined as follows:

  .. code:: python

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='iqn',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether priority replay is used.
        priority=False,
        # (float) Reward's future discount factor, aka. gamma.
        model=dict(
            # (int) Number of quantiles of the
            # value distribution. Default to 32.
            num_quantiles=32,
            # (int) Length of embedding of the quantiles. Default to 128.
            quantile_embedding_dim=128,
        ),
        discount_factor=0.97,
        # (int) N-step reward for target q_value estimation
        nstep=1,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after each collection of the collector .
            # collect data -> update policy-> collect data -> ...
            update_per_collect=3,
            # The batch size for each training iteration.
            batch_size=64,
            learning_rate=0.001,
            # (float) The weight of L2 reg loss of the network parameters.
            weight_decay=0.0,
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
            # (int) Frequence of target network update.
            target_update_freq=100,
            # (bool) Whether ignore done(usually for max step termination env)
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) Only one of [n_sample, n_step, n_episode] shoule be set
            n_sample=8,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (bool) Whether to use hindsight experience replay
            her=False,
            her_strategy='future',
            her_replay_k=1,
        ),
        eval=dict(),
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # (str) Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                # (int) Decay steps(env step).
                decay=10000,
            ),
            replay_buffer=dict(
                # (int) Replay buffer size.
                replay_buffer_size=10000,
            )
        ),
    )

    The bellman updates of QRDQN is implemented as:

    .. code:: python

      def iqn_nstep_td_error(
              data: namedtuple,
              gamma: float,
              nstep: int = 1,
              kappa: float = 1.0,
      ) -> torch.Tensor:
          """
          Overview:
              Multistep (1 step or n step) td_error with in IQN, \
                  referenced paper Implicit Quantile Networks for Distributional Reinforcement Learning \
                  <https://arxiv.org/pdf/1806.06923.pdf>
          Arguments:
              - data (:obj:`iqn_nstep_td_data`): the input data, iqn_nstep_td_data to calculate loss
              - gamma (:obj:`float`): discount factor
              - nstep (:obj:`int`): nstep num, default set to 1
              - criterion (:obj:`torch.nn.modules`): loss function criterion
              - beta_function (:obj:`Callable`): the risk function
          Returns:
              - loss (:obj:`torch.Tensor`): nstep td error, 0-dim tensor
          Shapes:
              - data (:obj:`q_nstep_td_data`): the q_nstep_td_data containing\
              ['q', 'next_n_q', 'action', 'reward', 'done']
              - q (:obj:`torch.FloatTensor`): :math:`(tau, B, N)` i.e. [tau x batch_size, action_dim]
              - next_n_q (:obj:`torch.FloatTensor`): :math:`(tau', B, N)`
              - action (:obj:`torch.LongTensor`): :math:`(B, )`
              - next_n_action (:obj:`torch.LongTensor`): :math:`(B, )`
              - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is timestep(nstep)
              - done (:obj:`torch.BoolTensor`) :math:`(B, )`, whether done in last timestep
          """
          q, next_n_q, action, next_n_action, reward, done, replay_quantiles, weight = data

          assert len(action.shape) == 1, action.shape
          assert len(next_n_action.shape) == 1, next_n_action.shape
          assert len(done.shape) == 1, done.shape
          assert len(q.shape) == 3, q.shape
          assert len(next_n_q.shape) == 3, next_n_q.shape
          assert len(reward.shape) == 2, reward.shape

          if weight is None:
              weight = torch.ones_like(action)

          batch_size = done.shape[0]
          tau = q.shape[0]
          tau_prime = next_n_q.shape[0]

          # shape: tau x batch_size x 1
          action = action.repeat([tau, 1]).unsqueeze(-1)
          # shape: tau x batch_size x 1
          next_n_action = next_n_action.repeat([tau_prime, 1]).unsqueeze(-1)

          # shape: batch_size x tau x 1
          q_s_a = torch.gather(q, -1, action).permute([1, 0, 2])
          # shape: batch_size x tau_prim x 1
          target_q_s_a = torch.gather(next_n_q, -1, next_n_action).permute([1, 0, 2])

          assert reward.shape[0] == nstep
          device = torch.device("cuda" if reward.is_cuda else "cpu")
          reward_factor = torch.ones(nstep).to(device)
          for i in range(1, nstep):
              reward_factor[i] = gamma * reward_factor[i - 1]
          # shape: batch_size
          reward = torch.matmul(reward_factor, reward)
          target_q_s_a = reward.unsqueeze(-1) + (gamma ** nstep) * target_q_s_a.squeeze(-1) * (1 - done).unsqueeze(-1)
          # shape: batch_size x tau_prim x 1
          target_q_s_a = target_q_s_a.unsqueeze(-1)

          # shape: batch_size x tau_prim x tau x 1.
          bellman_errors = (target_q_s_a[:, :, None, :] - q_s_a[:, None, :, :])

          # The huber loss (see Section 2.3 of the paper) is defined via two cases:
          huber_loss = torch.where(
              bellman_errors.abs() <= kappa, 0.5 * bellman_errors ** 2, kappa * (bellman_errors.abs() - 0.5 * kappa)
          )

          # Reshape replay_quantiles to batch_size x num_tau_samples x 1
          replay_quantiles = replay_quantiles.reshape([tau, batch_size, 1]).permute([1, 0, 2])

          # shape: batch_size x tau_prim x tau x 1.
          replay_quantiles = replay_quantiles[:, None, :, :].repeat([1, tau_prime, 1, 1])

          # shape: batch_size x tau_prime x tau x 1.
          quantile_huber_loss = (torch.abs(replay_quantiles - ((bellman_errors < 0).float()).detach()) * huber_loss) / kappa

          # shape: batch_size
          loss = quantile_huber_loss.sum(dim=2).mean(dim=1)[:, 0]

          return (loss * weight).mean(), loss
