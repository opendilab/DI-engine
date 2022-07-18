
Multi-Discrete Example
============================================

gym uses ``multi-discrete`` to refer to describe environments which have multiple discrete action spaces, a simple example is shown as follows:

.. code:: python

    import gym
    from gym.spaces import Discrete, MultiDiscrete
    """
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    """

    # discrete action space env
    env = gym.make('PongNoFrameskip-v4')
    assert env.action_space == Discrete(6)
    # multi discrete action space
    md_space = MultiDiscrete([2, 3])  # 6 = 2 * 3

In this page, a simple case of multi-discrete environment along with the multi-discrete implementation of DQN is provided.

Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here we provide a toy case of multi-discrete environment, which is derived by factorizing the single action space of Atari games in to the Cartesian product of multiple action spaces, e.g. 6=2*3.

.. code:: python

    class MultiDiscreteEnv(gym.Wrapper):
        """Map the actions from the factorized action spaces to the original single action space.

        :param gym.Env env: the environment to wrap.
        :param list action_shape: dims of the the factorized action spaces.
        """

        def __init__(self, env, action_shape):
            super().__init__(env)
            self.action_shape = np.flip(np.cumprod(np.flip(np.array(action_shape))))

        def step(self, action):
            """
            Overview:
                Step the environment with the given factorized actions.
            Arguments:
                - action (:obj:`list`): a list contains the action output of each discrete dimension, e.g.: [1, 1] means 1 * 3 + 1 = 4 for a factorized action 2 * 3 = 6
            """
            action = action[0] * self.action_shape[1] + action[0]
            obs, reward, done, info = self.env.step(action)
            return obs, reward, done, info

Accordingly, the config of a multi-discrete experiment should be altered by changing the ``action_shape`` from an integer into the list of the dims of the factorized action spaces, which locates at ``config.policy.model`` and ``env.info()``. Also, the key ``multi_discrete`` in ``config.env`` should be set True to utilize the ``MultiDiscreteEnv`` wrapper.

Then we provide a multi-discrete version of DQN implementation. The multi-discrete version ``DQNMultiDiscretePolicy`` inherits ``DQNPolicy`` and only overrides the ``_forward_learn`` interface. In the Q-learning forward part of this overrode version, each action space calculates its own q-value, action and td loss with the global rewards, following the same process of the single action space.

.. code:: python

            # ====================
            # Q-learning forward
            # ====================
            self._learn_model.train()
            self._target_model.train()
            # Current q value (main model)
            q_value = self._learn_model.forward(data['obs'])['logit']
            # Target q value
            with torch.no_grad():
                target_q_value = self._target_model.forward(data['next_obs'])['logit']
                # Max q value action (main model)
                target_q_action = self._learn_model.forward(data['next_obs'])['action']

            value_gamma = data.get('value_gamma')
            if isinstance(q_value, torch.Tensor):
                data_n = q_nstep_td_data(
                    q_value, target_q_value, data['action'], target_q_action, data['reward'], data['done'], data['weight']
                )
                loss, td_error_per_sample = q_nstep_td_error(data_n, self._gamma, nstep=self._nstep, value_gamma=value_gamma)
            else:
                action_num = len(q_value)
                loss, td_error_per_sample = [], []
                for i in range(action_num):
                    td_data = q_nstep_td_data(
                        q_value[i], target_q_value[i], data['action'][i], target_q_action[i], data['reward'], data['done'], data['weight']
                    )
                    loss_, td_error_per_sample_ = q_nstep_td_error(td_data, self._gamma, nstep=self._nstep)
                    loss.append(loss_)
                    td_error_per_sample.append(td_error_per_sample_.abs())
                loss = sum(loss) / (len(loss) + 1e-8)
                td_error_per_sample = sum(td_error_per_sample) / (len(td_error_per_sample) + 1e-8)

For the complete code, you can refer to ``dizoo/common/policy/md_dqn.py``
