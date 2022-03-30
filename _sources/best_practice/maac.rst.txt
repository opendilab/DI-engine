Multi-Agent Actor-Critic RL
============================================
MARL algorithms can be divided into two broad categories: centralized learning and decentralized learning. Recent work has developed two lines of research to bridge the gap between these two frameworks: centralized training and decentralized execution(CTDE) and value decomposition(VD).
VD such as Qmix typically represents the joint Q-function as a function of agents’ local Q-functions, which has been considered as the gold standard for many MARL benchmarks.
CTDE methods such as MADDPG, MAPPO and COMA improve upon decentralized RL by adopting an actor-critic structure and learning a centralized critic. 
In DI-engine, we introduce the multi-agent actor-critic framework to quickly convert a single-agent algorithm into a multi-agent algorithm.


For Users
--------------------------

Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Unlike single-agent environments that return a tensor-type observation, our multi-agent environments will return a dict-type observation, which includes ``agent_state``, ``global_state`` and ``action_mask``.

.. code:: python 

   agent_num = 8
   agent_obs_shape = 150
   global_obs_shape = 295
   action_shape = 14
   return {
         'agent_state': torch.randn(agent_num, agent_obs_shape),
         'global_state': torch.randn(agent_num, global_obs_shape),
         'action_mask': torch.randint(0, 2, size=(agent_num, action_shape))
   }

- agent state: An agent state comprises of each agent's local observation.
- global state: A global state contains all global information that can't be seen by each agent.
- action mask: In multi-agent games, it is often the case that some actions cannot be executed due to game constraints. For example, in SMAC, an agent may have skills that cannot be performed frequently. So, when computing the logits for the softmax action probability, we mask out the unavailable actions in both the forward and backward pass so that the probabilities for unavailable actions are always zero. We find that this substantially accelerates training. The data type is \ ``int``\.
- death mask: In multi-agent games, an agent may die before the game terminates, such as SAMC environment. Note that we can always access the game state to compute the agent-specific global state for those dead agents. Therefore, even if an agent dies and becomes inactive in the middle of a rollout, value learning can still be performed in the following timesteps using inputs containing information of other live agents. This is typical in many existing multi-agent PG implementations. Our suggestion is to simply use a zero vector with the agent’s ID as the input to the value function after an agent dies. We call this approach “Death Masking”. The idea was proposed in `The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games <https://arxiv.org/abs/2103.01955>`_

In our environments, it can return four different global states, they have different uses.

- global obs: It contains all global information, default to return it.
- agent specific global obs: Global observation that contains all global information and the necessary agent-specific features, such as agent id, available actions. If you want to use it, you have to set ``special_global_state`` to ``True`` in env config.
- collaq obs: It contains agent_alone_state and agent_alone_padding_state, you can use it in Collaq alg. Agents_obs_alone means the agent can't observe the allies' information, agent_alone_padding_state means the agent's allies' information is zero. If you want to use it, you have to set ``obs_alone`` to ``True`` in env config.
- independent obs: The global observation is as same as agent observation, we use it in independent PPO, independent SAC alg. If you want to use it, you have to set ``independent_obs`` to ``True`` in env config.

Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Centralized training and decentralized executed: Unlike single-agent environments that feed the same observation information to actor and critic networks, in multi-agent environments, we feed agent_state and action_mask information to the actor network to get each actions' logits and mask the invalid/inaccessible actions. At the same time, we feed global_state information to the critic network to gei the global critic value.
- Action mask: We need to mask the invalid/inaccessible actions when we train or collect data. So we use ``logit[action_mask == 0.0] = -99999999`` to make the inaccessible actions' probability to a very low value. So we can't choose this action when we collect data or train the model. If you don't want to use it, just delete ``logit[action_mask == 0.0] = -99999999``.

.. code:: python 

    def compute_actor(self, x: torch.Tensor) -> Dict:
        action_mask = x['action_mask']
        x = x['agent_state']
        x = self.actor_encoder(x)
        x = self.actor_head(x)
        logit = x['logit']
        # action mask
        logit[action_mask == 0.0] = -99999999
        return {'logit': logit}

    def compute_critic(self, x: Dict) -> Dict:
        x = self.critic_encoder(x['global_state'])
        x = self.critic_head(x)
        return {'value': x['pred']}

Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When modifying the single-agent algorithm into a multi-agent algorithm, the policy part basically remains the same, the only thing to note is to add the multi_agent key in the config and it will call the multi-agent model when the multi_agent key is True.

When you use the single-agent algorithm, ``multi_agent`` is default to ``False``, you don't need to do anything. And when you use the multi-agent algorithm, you have to add the ``multi_agent`` key and set it to ``True``.



Config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Open the multi-agent key and just change the environment to the one you want to run. 

.. code:: python 

   agent_num = 5
   collector_env_num = 8
   evaluator_env_num = 8
   special_global_state = True,

   main_config = dict(
      exp_name='smac_5m6m_ppo',
      env=dict(
         map_name='5m_vs_6m',
         difficulty=7,
         reward_only_positive=True,
         mirror_opponent=False,
         agent_num=agent_num,
         collector_env_num=collector_env_num,
         evaluator_env_num=evaluator_env_num,
         n_evaluator_episode=16,
         stop_value=0.99,
         death_mask=True,
         special_global_state=special_global_state,
         manager=dict(
               shared_memory=False,
               reset_timeout=6000,
         ),
      ),
      policy=dict(
         cuda=True,
         multi_agent=True,
         continuous=False,
         model=dict(
               # (int) agent_num: The number of the agent.
               # For SMAC 3s5z, agent_num=8; for 2c_vs_64zg, agent_num=2.
               agent_num=agent_num,
               # (int) obs_shape: The shapeension of observation of each agent.
               # For 3s5z, obs_shape=150; for 2c_vs_64zg, agent_num=404.
               # (int) global_obs_shape: The shapeension of global observation.
               # For 3s5z, obs_shape=216; for 2c_vs_64zg, agent_num=342.
               agent_obs_shape=72,
               #global_obs_shape=216,
               global_obs_shape=152,
               # (int) action_shape: The number of action which each agent can take.
               # action_shape= the number of common action (6) + the number of enemies.
               # For 3s5z, obs_shape=14 (6+8); for 2c_vs_64zg, agent_num=70 (6+64).
               action_shape=12,
               # (List[int]) The size of hidden layer
               # hidden_size_list=[64],
         ),
         # used in state_num of hidden_state
         learn=dict(
               # (bool) Whether to use multi gpu
               multi_gpu=False,
               epoch_per_collect=10,
               batch_size=3200,
               learning_rate=5e-4,
               # ==============================================================
               # The following configs is algorithm-specific
               # ==============================================================
               # (float) The loss weight of value network, policy network weight is set to 1
               value_weight=0.5,
               # (float) The loss weight of entropy regularization, policy network weight is set to 1
               entropy_weight=0.01,
               # (float) PPO clip ratio, defaults to 0.2
               clip_ratio=0.05,
               # (bool) Whether to use advantage norm in a whole training batch
               adv_norm=False,
               value_norm=True,
               ppo_param_init=True,
               grad_clip_type='clip_norm',
               grad_clip_value=10,
               ignore_done=False,
         ),
         on_policy=True,
         collect=dict(env_num=collector_env_num, n_sample=3200),
         eval=dict(env_num=evaluator_env_num),
      ),
   )
   main_config = EasyDict(main_config)
   create_config = dict(
      env=dict(
         type='smac',
         import_names=['dizoo.smac.envs.smac_env'],
      ),
      env_manager=dict(type='base'),
      policy=dict(type='ppo'),
   )
   create_config = EasyDict(create_config)


The following are the parameters for each map of the SMAC environment.

+------------------+---------------------+--------------------+--------------------------------+---------------------+
| Map              | agent_obs_shape     | global_obs_shape   | agent_special_global_obs_shape | action_shape        |
+==================+=====================+====================+================================+=====================+
| 3s5z             | 150                 | 216                |        295                     | 14                  |
+------------------+---------------------+--------------------+--------------------------------+---------------------+
| 5m_vs_6m         | 72                  | 98                 |        152                     | 12                  |
+------------------+---------------------+--------------------+--------------------------------+---------------------+
| MMM              | 186                 | 290                |        389                     | 16                  |
+------------------+---------------------+--------------------+--------------------------------+---------------------+
| MMM2             | 204                 | 322                |        431                     | 18                  |
+------------------+---------------------+--------------------+--------------------------------+---------------------+
| 2c_vs_64zg       | 404                 | -                  |        671                     | 70                  |
+------------------+---------------------+--------------------+--------------------------------+---------------------+
| 6h_vs_8z         | 98                  | -                  |        209                     | 14                  |
+------------------+---------------------+--------------------+--------------------------------+---------------------+
| 3s5z_vs_3s6z     | 159                 | -                  |        314                     | 15                  |
+------------------+---------------------+--------------------+--------------------------------+---------------------+
| 25m              | 306                 | -                  |        1199                    | 31                  |
+------------------+---------------------+--------------------+--------------------------------+---------------------+
| 8m_vs_9m         | 108                 | -                  |        263                     | 15                  |
+------------------+---------------------+--------------------+--------------------------------+---------------------+
| 10m_vs_11m       | 132                 | -                  |        347                     | 17                  |
+------------------+---------------------+--------------------+--------------------------------+---------------------+
| 27m_vs_30m       | 348                 | -                  |        1454                    | 36                  |
+------------------+---------------------+--------------------+--------------------------------+---------------------+
| corridor         | 192                 | -                  |        431                     | 30                  |
+------------------+---------------------+--------------------+--------------------------------+---------------------+

-  SMAC environment 3s5z map training performance

   - 3s5z + MAPPO/IPPO

   .. image:: images/3s5z_mappo.png
     :align: center

For Developers
--------------------------

Model
^^^^^^^^^^^^^^^^^^
We need to change the single agent to the multi agent model. In single agent model, it only has a obs_shape key. In multi agent model, we need to divide the obs_shape key to agent_obs_shape and global_obs_shape, and in this way, we can train critic model by global obs and train actor model by agent obs.

Policy
^^^^^^^^^^^^^^^^^^
We need to call the multi agent model in the following way.

.. code:: python 

    MAPPO:

    def default_model(self) -> Tuple[str, List[str]]:
        if self._cfg.multi_agent:
            return 'mappo', ['ding.model.template.mappo']
        else:
            return 'vac', ['ding.model.template.vac']

    MASAC:

    def default_model(self) -> Tuple[str, List[str]]:
        if self._cfg.multi_agent:
            return 'maqac', ['ding.model.template.maqac']
        else:
            return 'qac', ['ding.model.template.qac']

rl_utils
^^^^^^^^^^^^^^^^^^
In the single agent algorithm, the data dimension is (B, N), the B means batch_size, and the N means the action nums. But in the multi agent algorithm, the data dimension is (B, A, N), the A means action nums. So when we calculate the loss function, we need to change our codes.
For example, when we calculate the PPO advantage, we need to modify the codes. For most time, we use unsqueeze to change the (B, N) to (B, 1, N), and it can operate with (B, A, N) data.


.. code:: python 

    def gae(data: namedtuple, gamma: float = 0.99, lambda_: float = 0.97) -> torch.FloatTensor:
        """
        Overview:
            Implementation of Generalized Advantage Estimator
        """
        value, next_value, reward, done, traj_flag = data
        if done is None:
            done = torch.zeros_like(reward, device=reward.device)

        # In Multi-agent RL, the value and next_value's dimension is (B, A), the reward and done's dimension is (B) not (B,N), we unsqueeze the reward and done to change their shape from (B) to (B, 1).
        if len(value.shape) == len(reward.shape) + 1:
            reward = reward.unsqueeze(-1)
            done = done.unsqueeze(-1)
        delta = reward + (1 - done) * gamma * next_value - value
        factor = gamma * lambda_
        adv = torch.zeros_like(value, device=value.device)
        gae_item = torch.zeros_like(value[0])

        for t in reversed(range(reward.shape[0])):
            if traj_flag is None:
                gae_item = delta[t] + factor * gae_item * (1 - done[t])
            else:
                gae_item = delta[t] + factor * gae_item * (1 - traj_flag[t].float())
            adv[t] += gae_item
        return adv

When we change the code, we need to test our codes by the following way.
You can just input (B, N) data to test single agent rl utils codes and input (B, A, N) data to test multi agent rl utils codes.

.. code:: python

    def test_ppo():
        B, N = 4, 32
        logit_new = torch.randn(B, N).requires_grad_(True)
        logit_old = logit_new + torch.rand_like(logit_new) * 0.1
        action = torch.randint(0, N, size=(B, ))
        value_new = torch.randn(B).requires_grad_(True)
        value_old = value_new + torch.rand_like(value_new) * 0.1
        adv = torch.rand(B)
        return_ = torch.randn(B) * 2
        data = ppo_data(logit_new, logit_old, action, value_new, value_old, adv, return_)
        loss, info = ppo_error(data)
        assert all([l.shape == tuple() for l in loss])
        assert all([np.isscalar(i) for i in info])
        assert logit_new.grad is None
        assert value_new.grad is None
        total_loss = sum(loss)
        total_loss.backward()
        assert isinstance(logit_new.grad, torch.Tensor)
        assert isinstance(value_new.grad, torch.Tensor)

    def test_mappo():
        B, A, N = 4, 8, 32
        logit_new = torch.randn(B, A, N).requires_grad_(True)
        logit_old = logit_new + torch.rand_like(logit_new) * 0.1
        action = torch.randint(0, N, size=(B, A))
        value_new = torch.randn(B, A).requires_grad_(True)
        value_old = value_new + torch.rand_like(value_new) * 0.1
        adv = torch.rand(B, A)
        return_ = torch.randn(B, A) * 2
        data = ppo_data(logit_new, logit_old, action, value_new, value_old, adv, return_, None)
        loss, info = ppo_error(data)
        assert all([l.shape == tuple() for l in loss])
        assert all([np.isscalar(i) for i in info])
        assert logit_new.grad is None
        assert value_new.grad is None
        total_loss = sum(loss)
        total_loss.backward()
        assert isinstance(logit_new.grad, torch.Tensor)
        assert isinstance(value_new.grad, torch.Tensor)
