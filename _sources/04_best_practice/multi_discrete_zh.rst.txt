
多重离散动作空间示例
============================================

多重离散动作空间指多维度的离散动作空间，可以理解为是离散动作空间的n维形式。比如我们每次执行的动作有n个维度，每个维度都由一个离散动作空间构成。

gym库使用 ``多重离散（multi-discrete）`` 去描述具有多个离散动作空间的环境。一个简单的例子如下所示：

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

下面，我们提供了一个简单的多重离散动作空间环境的例子以及如何将 DQN 算法变为 DQN 算法的多重离散动作空间实现，并应用到该环境上。

举例
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
在这里，我们提供了一个多重离散动作空间环境的案例，它是通过将 Atari 游戏的单个动作空间分解为多个动作空间的笛卡尔积而得出的，例如6=2*3。

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
            action = action[0] * self.action_shape[1] + action[1]
            obs, reward, done, info = self.env.step(action)
            return obs, reward, done, info

因此，多重离散动作空间的实验的配置应该通过将 ``action_shape`` 从一个整数更改为通过分解动作空间维度而形成的列表来改变, 这位于 ``config.policy.model`` 和 ``env.info()``。

此外，在 config.env 中的键 multi_discrete 应该设置为 True 以使用 MultiDiscreteEnv wrapper。

我们也提供了一个多离散版本的 DQN 实现. 多重离散版本``DQNMultiDiscretePolicy``继承``DQNPolicy``并且只覆盖``_forward_learn``接口. 在这个多重离散动作空间版本的 Q-learning 的 ``_forward_learn`` 中，每层动作空间使用全局奖励计算自己的 q 值、动作和 td 损失，计算遵循与单个动作空间计算相同的过程。

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

完整代码可以参考 ``dizoo/common/policy/md_dqn.py``
