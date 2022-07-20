PPO
^^^^^^^

Overview
---------
PPO (Proximal Policy Optimization) was proposed in `Proximal Policy Optimization Algorithms <https://arxiv.org/pdf/1707.06347.pdf>`_.
The key question to answer is that how can we utilize the existing data to take the most possible improvement step for the policy
without accidentally leading to performance collapse.
PPO follows the idea of TRPO (which restricts the step of policy update by explicit KL-divergence constraint),
but doesn’t have a KL-divergence term in the objective,
instead utilizing a specialized clipped objective to remove incentives for the new policy to get far from the old policy.
PPO avoids the calculation of the Hessian matrix in TRPO, thus is simpler to implement and empirically performs at least as well as TRPO.

Quick Facts
-----------
1. PPO is a **model-free** and **policy-gradient** RL algorithm.

2. PPO supports both **discrete** and **continuous action spaces**.

3. PPO supports **off-policy** mode and **on-policy** mode.

4. PPO can be equipped with RNN.

5. PPO is a first-order gradient method that use a few tricks to keep new policies close to old.

Key Equations or Key Graphs
------------------------------
PPO use clipped probability ratios in the policy gradient to prevent the policy from too rapid changes, specifically the
optimizing objective is:

.. math::
    L_{\theta_{k}}^{C L I P}(\theta) \doteq {\mathrm{E}}_{s, a \sim \theta_{k}}\left[\min \left(\frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{k}}(a \mid s)} A^{\theta_{k}}(s, a), {clip}\left(\frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{k}}(a \mid s)}, 1-\epsilon, 1+\epsilon\right) A^{\theta_{k}}(s, a)\right)\right]

where :math:`\frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{k}}(a \mid s)}` is denoted as the probability ratio :math:`r_t(\theta)`,
:math:`\theta` are the policy parameters to be optimized at the current time, :math:`\theta_k` are the parameters of the policy at iteration k and :math:`\gamma` is a small hyperparameter control that controls the maximum update step size of the policy parameters.

..
    .. math::
        r_{t}(\theta)=\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)}
    When :math:`\hat{A}_t > 0`, :math:`r_t(\theta) > 1 + \epsilon` will be clipped. While when :math:`\hat{A}_t < 0`, :math:`r_t(\theta) < 1 - \epsilon` will be clipped.

According to this `note <https://drive.google.com/file/d/1PDzn9RPvaXjJFZkGeapMHbHGiWWW20Ey/view?usp=sharing>`__, the PPO-Clip objective can be simplified to:

.. math::
    L_{\theta_{k}}^{C L I P}(\theta)={\mathrm{E}}_{s, a \sim \theta_{k}}\left[\min \left(\frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{k}}(a \mid s)} A^{\theta_{k}}(s, a), g\left(\epsilon, A^{\theta_{k}}(s, a)\right)\right)\right]

where,

.. math::
    g(\epsilon, A)= \begin{cases}(1+\epsilon) A & A \geq 0 \\ (1-\epsilon) A & \text { otherwise }\end{cases}

Usually we don't access to the true advantage value of the sampled state-action pair :math:`(s,a)`, but luckily we can calculate a approximate value :math:`\hat{A}_t`.
The idea behind this clipping objective is: for :math:`(s,a)`, if :math:`\hat{A}_t < 0`, maximizing :math:`L^{C L I P}(\theta)` means make :math:`\pi_{\theta}(a_{t} \mid s_{t})` smaller, but no additional benefit to the objective function is gained
by making :math:`\pi_{\theta}(a_{t} \mid s_{t})` smaller than :math:`(1-\epsilon)\pi_{\theta}(a_{t} \mid s_{t})`
. Analogously, if :math:`\hat{A}_t > 0`, maximizing :math:`L^{C L I P}(\theta)` means make :math:`\pi_{\theta}(a_{t} \mid s_{t})` larger, but no additional benefit is gained by making :math:`\pi_{\theta}(a_{t} \mid s_{t})`
larger than :math:`(1+\epsilon)\pi_{\theta}(a_{t} \mid s_{t})`.
Empirically, by optimizing this objective function, the update step of the policy network can be controlled within a reasonable range.

For the value function, in order to balance the bias and variance in value learning, PPO adopts the `Generalized Advantage Estimator <https://arxiv.org/abs/1506.02438>`__ to compute the advantages,
which is a exponentially-weighted sum of Bellman residual terms.  that is analogous to TD(λ):

.. math::
    \hat{A}_{t}=\delta_{t}+(\gamma \lambda) \delta_{t+1}+\cdots+\cdots+(\gamma \lambda)^{T-t+1} \delta_{T-1}

where V is an approximate value function, :math:`\delta_{t}=r_{t}+\gamma V\left(s_{t+1}\right)-V\left(s_{t}\right)` is the Bellman residual terms, or called TD-error at timestep t.

The value target is calculated as: :math:`V_{t}^{target}=V_{t}+\hat{A}_{t}`,
and the value loss is defined as a squared-error: :math:`\frac{1}{2}*\left(V_{\theta}\left(s_{t}\right)-V_{t}^{\mathrm{target}}\right)^{2}`,
To ensure adequate exploration, PPO further enhances the objective by adding a policy entropy bonus.

The total PPO loss is a weighted sum of policy loss, value loss and policy entropy regularization term:

.. math::
    L_{t}^{total}=\hat{\mathbb{E}}_{t}[ L_{t}^{C L I P}(\theta)+c_{1} L_{t}^{V F}(\phi)-c_{2} H(a_t|s_{t}; \pi_{\theta})]

where c1 and c2 are coefficients that control the relative importance of different terms.

.. note::
    The standard implementation of PPO contains the many additional optimizations which are not described in the paper. Further details can be found in `IMPLEMENTATION MATTERS IN DEEP POLICY GRADIENTS: A CASE STUDY ON PPO AND TRPO <https://arxiv.org/abs/2005.12729>`_.

Pseudo-code
-----------
.. image:: images/PPO_onpolicy.png
   :align: center
   :scale: 50%

.. note::
    This is the on-policy version of PPO. In DI-engine, we also have the off-policy version of PPO, which is almost the same as on-policy PPO except that
    we maintain a replay buffer that stored the recent experience,
    and the data used to calculate the PPO loss is sampled from the replay buffer not the recently collected batch,
    so off-policy PPO are able to reuse old data very efficiently, but potentially brittle and unstable.


Extensions
-----------

PPO can be combined with:
    - `Multi-step learning <https://di-engine-docs.readthedocs.io/en/latest/best_practice/nstep_td.html>`__
    - `RNN <https://di-engine-docs.readthedocs.io/en/latest/best_practice/rnn.html>`__


Implementation
-----------------
The default config is defined as follows:

    .. autoclass:: ding.policy.ppo.PPOPolicy


    .. autoclass:: ding.model.template.vac.VAC
        :members: forward, compute_actor, compute_critic, compute_actor_critic
        :noindex:


The policy loss and value loss of PPO is implemented as follows:

.. code:: python

    def ppo_error(
            data: namedtuple,
            clip_ratio: float = 0.2,
            use_value_clip: bool = True,
            dual_clip: Optional[float] = None
    ) -> Tuple[namedtuple, namedtuple]:

        assert dual_clip is None or dual_clip > 1.0, "dual_clip value must be greater than 1.0, but get value: {}".format(
            dual_clip
        )
        logit_new, logit_old, action, value_new, value_old, adv, return_, weight = data
        policy_data = ppo_policy_data(logit_new, logit_old, action, adv, weight)
        policy_output, policy_info = ppo_policy_error(policy_data, clip_ratio, dual_clip)
        value_data = ppo_value_data(value_new, value_old, return_, weight)
        value_loss = ppo_value_error(value_data, clip_ratio, use_value_clip)

        return ppo_loss(policy_output.policy_loss, value_loss, policy_output.entropy_loss), policy_info

The interface of ``ppo_policy_error`` and ``ppo_value_error`` is defined as follows:

    .. autofunction:: ding.rl_utils.ppo.ppo_policy_error

    .. autofunction:: ding.rl_utils.ppo.ppo_value_error


Implementation Tricks
-----------------------

.. list-table:: Some Implementation Tricks that Matter
   :widths: 25 15
   :header-rows: 1

   * - trick
     - explanation
   * - | `Generalized Advantage Estimator <https://github.com/opendilab/DI-engine/blob/e89d8fdc4b7340c708b48f987a8e9f312cd0f7a2/ding/rl_utils/gae.py#L26>`__
     - | Utilizing generalized advantage estimator to balance bias and variance in value learning.
   * - | `Dual Clip <https://github.com/opendilab/DI-engine/blob/7630dbaa65e4ef33b07cc0f6c630fce280aa200c/ding/rl_utils/ppo.py#L193>`__
     - | In the paper `Mastering Complex Control in MOBA Games with Deep Reinforcement Learning <https://arxiv.org/abs/1912.09729>`_,
       | the authors claim that when :math:`\hat{A}_t < 0`, a too large :math:`r_t(\theta)` should also be clipped, which introduces dual clip:
       | :math:`\max \left(\min \left(r_{t}(\theta) \hat{A}_{t}, {clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}\right), c \hat{A}_{t}\right)`
   * - | `Recompute Advantage <https://github.com/opendilab/DI-engine/blob/7630dbaa65e4ef33b07cc0f6c630fce280aa200c/ding/policy/ppo.py#L171>`__
     - | In on-policy PPO, each time we collect a batch data, we will train many epochs to improve data efficiency.
       | And before the beginning of each training epoch, we recompute the advantage of historical transitions,
       | to keep the advantage is an approximate evaluation of current policy.
   * - | `Value/Advantage Normalization <https://github.com/opendilab/DI-engine/blob/7630dbaa65e4ef33b07cc0f6c630fce280aa200c/ding/policy/ppo.py#L175>`__
     - | We standardize the targets of the value/advantage function using running estimates of the average
       | and standard deviation of the value/advantage targets. For more implementation details about
       | recompute advantage and normalization, users can refer to this `discussion <https://github.com/opendilab/DI-engine/discussions/172#discussioncomment-1901038>`__.
   * - | `Value Clipping <https://github.com/opendilab/DI-engine/blob/e6cc06043b479b164b41189ac99c9315c0c938de/ding/rl_utils/ppo.py#L202>`_
     - | Value is clipped around the previous value estimates. We use the value clip_ratio same as that used to clip policy
       | probability ratios in the PPO policy loss function.
   * - | `Orthogonal initialization <https://github.com/opendilab/DI-engine/blob/7630dbaa65e4ef33b07cc0f6c630fce280aa200c/ding/policy/ppo.py#L98>`__
     - | Using an orthogonal initialization scheme for the policy and value networks.

..
    .. code:: python
        if use_value_clip:
            value_clip = value_old + (value_new - value_old).clamp(-clip_ratio, clip_ratio)
            v1 = (return_ - value_new).pow(2)
            v2 = (return_ - value_clip).pow(2)
            value_loss = 0.5 * (torch.max(v1, v2) * weight).mean()

..


Benchmark
-----------

off-policy PPO Benchmark:


+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| environment         |best mean reward | evaluation results                                  | config link              | comparison           |
+=====================+=================+=====================================================+==========================+======================+
|                     |                 |                                                     |`config_link_p <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|                     |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|Pong                 |  20             |.. image:: images/benchmark/pong_offppo.png          |atari/config/serial/      |                      |
|                     |                 |                                                     |pong/pong_offppo_config   |                      |
|(PongNoFrameskip-v4) |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_q <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|Qbert                |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |  16400          |.. image:: images/benchmark/qbert_offppo.png         |atari/config/serial/      |                      |
|(QbertNoFrameskip-v4)|                 |                                                     |qbert/qbert_offppo_config |                      |
|                     |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_s <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|SpaceInvaders        |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |  1200           |.. image:: images/benchmark/spaceinvaders_offppo.png |atari/config/serial/      |                      |
|(SpaceInvadersNoFrame|                 |                                                     |spaceinvaders/spaceinva   |                      |
|skip-v4)             |                 |                                                     |ders_offppo_config.py>`_  |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_ho <https:// |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|                     |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|Hopper               |  300            |.. image:: images/benchmark/hopper_offppo.png        |mujoco/config/hopper_     |                      |
|                     |                 |                                                     |offppo_default_config     |                      |
|(Hopper-v3)          |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_w <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|Walker2d             |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |  500            |.. image:: images/benchmark/walker2d_offppo.png      |mujoco/config/            |                      |
|(Walker2d-v3)        |                 |                                                     |walker2d_offppo_          |                      |
|                     |                 |                                                     |default_config.py>`_      |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_ha <https:// |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|Halfcheetah          |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |  2000           |.. image:: images/benchmark/halfcheetah_offppo.png   |mujoco/config/            |                      |
|(Halfcheetah-v3)     |                 |                                                     |halfcheetah_offppo        |                      |
|                     |                 |                                                     |_default_config.py>`_     |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+


on-policy PPO Benchmark:


+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| environment         |best mean reward | evaluation results                                  | config link              | comparison           |
+=====================+=================+=====================================================+==========================+======================+
|                     |                 |                                                     |`config_link_p <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|                     |                 |                                                     |DI-engine/tree/main/dizoo/|    RLlib(20)         |
|Pong                 |  20             |.. image:: images/benchmark/pong_onppo.png           |atari/config/serial/      |                      |
|                     |                 |                                                     |pong/pong_onppo_config    |                      |
|(PongNoFrameskip-v4) |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_q <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|Qbert                |                 |                                                     |DI-engine/tree/main/dizoo/|    RLlib(11085)      |
|                     |  10000          |.. image:: images/benchmark/qbert_onppo.png          |atari/config/serial/      |                      |
|(QbertNoFrameskip-v4)|                 |                                                     |qbert/qbert_onppo_config  |                      |
|                     |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_s <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|SpaceInvaders        |                 |                                                     |DI-engine/tree/main/dizoo/|    RLlib(671)        |
|                     |  800            |.. image:: images/benchmark/spaceinvaders_onppo.png  |atari/config/serial/      |                      |
|(SpaceInvadersNoFrame|                 |                                                     |spaceinvaders/spacein     |                      |
|skip-v4)             |                 |                                                     |vaders_onppo_config.py>`_ |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_ho <https:// |    Tianshou(3127)    |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|                     |                 |                                                     |DI-engine/tree/main/dizoo/|       Sb3(1567)      |
|Hopper               |  3000           |.. image:: images/benchmark/hopper_onppo.png         |mujoco/config/            |                      |
|                     |                 |                                                     |hopper_onppo_             |    spinningup(2500)  |
|(Hopper-v3)          |                 |                                                     |default_config.py>`_      |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_w <https://  |    Tianshou(4895)    |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|Walker2d             |                 |                                                     |DI-engine/tree/main/dizoo/|     Sb3(1230)        |
|                     |  3000           |.. image:: images/benchmark/walker2d_onppo.png       |mujoco/config/            |                      |
|(Walker2d-v3)        |                 |                                                     |walker2d_onppo_           |    spinningup(2500)  |
|                     |                 |                                                     |default_config.py>`_      |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_ha <https:// |    Tianshou(7337)    |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|Halfcheetah          |                 |                                                     |DI-engine/tree/main/dizoo/|     Sb3(1976)        |
|                     |  3500           |.. image:: images/benchmark/halfcheetah_onppo.png    |mujoco/config/            |                      |
|(Halfcheetah-v3)     |                 |                                                     |halfcheetah_onppo         |   spinningup(3000)   |
|                     |                 |                                                     |_default_config.py>`_     |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+


References
-----------

- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov: “Proximal Policy Optimization Algorithms”, 2017; [http://arxiv.org/abs/1707.06347 arXiv:1707.06347].

- Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph, Aleksander Madry: “Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO”, 2020; [http://arxiv.org/abs/2005.12729 arXiv:2005.12729].

- Andrychowicz M, Raichuk A, Stańczyk P, et al. What matters in on-policy reinforcement learning? a large-scale empirical study[J]. arXiv preprint arXiv:2006.05990, 2020.

- Ye D, Liu Z, Sun M, et al. Mastering complex control in moba games with deep reinforcement learning[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2020, 34(04): 6672-6679.

- https://spinningup.openai.com/en/latest/algorithms/ppo.html

Other Public Implementations
----------------------------

- spinningup_
- `RLlib (Ray)`_
- `SB3 (StableBaselines3)`_
- Tianshou_

.. _spinningup: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
.. _`RLlib (Ray)`: https://github.com/ray-project/ray/tree/master/python/ray/rllib/agents/ppo
.. _`SB3 (StableBaselines3)`: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
.. _Tianshou: https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/ppo.py
