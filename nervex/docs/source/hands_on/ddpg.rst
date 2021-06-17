DDPG
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview
---------

Deep Deterministic Policy Gradient (DDPG), proposed in the 2015 paper `Continuous control with deep reinforcement learning <https://arxiv.org/abs/1509.02971>`_, is an algorithm which learns a Q-function and a policy simultaneously.
DDPG is an actor-critic, model-free algorithm based on the deterministic policy gradient(DPG) that can operate over continuous action spaces.
DPG `Deterministic policy gradient algorithms <http://proceedings.mlr.press/v32/silver14.pdf>`_ algorithm is similar to NFQCA.

Quick Facts
-----------
1. DDPG is only used for environments with **continuous action spaces**.(i.e. MuJoCo)

2. DDPG is an **off-policy** algorithm.

3. DDPG is a **model-free** and **actor-critic** RL algorithm, which optimizes actor network and critic network, respectively.

4. Usually, DDPG use **Ornstein-Uhlenbeck process** or **Gaussian process** (default in our implementation) for exploration.

Key Equations or Key Graphs
---------------------------
The exploration policy by adding noise sampled from a noise process N to actor policy:

.. math::
    \mu^{\prime}\left(s_{t}\right)=\mu\left(s_{t} \mid \theta_{t}^{\mu}\right)+\mathcal{N}


Pseudocode
----------

.. image:: images/DDPG.jpg

.. note::
   Compared with the vanilla version, DDPG has been dramatically improved in implementation. We use Gaussian process noise for exploration.

Extensions
-----------
DDPG can be combined with:
    - Target Network
    - Replay Buffers.
    - Gaussian noise during collecting transition.



Implementations
----------------
The default config is defined as follows:

.. autoclass:: nervex.policy.ddpg.DDPGPolicy

The Benchmark result of DDPG implemented in nerveX is shown in `Benchmark <../feature/algorithm_overview.html>`_

Model
---------------------------------
Here we provide examples of `QAC` model as default model for `DDPG`.

QAC
~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.model.qac.q_ac.QAC
    :members: __init__, forward, seed, optimize_actor, compute_q, compute_action, mimic

Policy
---------------------------------

Train actor-critic model
~~~~~~~~~~~~~~~~~~~~~~~~~

The learn process in nerveX can be customized arbitrarily.
Usually the learn process in actor-critic algorithm consists of computing critic loss, updating critic network, computing actor loss, and updating actor network.

Here we provide examples of `DDPG` for `MuJoCo` environment.

First, we initialize actor and critic optimizer in ``_init_learn``, respectively.

.. code-block:: python

    # actor and critic optimizer
    self._optimizer_actor = Adam(
        self._model.actor.parameters(),
        lr=self._cfg.learn.learning_rate_actor,
    )
    self._optimizer_critic = Adam(
        self._model.critic.parameters(),
        lr=self._cfg.learn.learning_rate_critic,
    )

In ``learn`` we update actor-critic policy through computing critic loss, updating critic network, computing actor loss, and updating actor network.
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
