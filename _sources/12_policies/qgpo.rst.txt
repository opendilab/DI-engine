QGPO
^^^^^^^

Overview
---------

Q-Guided Policy Optimization(QGPO), proposed in the 2023 paper `Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning <https://arxiv.org/abs/2304.12824>`_, is an actor-critic offline RL algorithm based on energy-based conditional diffusion model.

Three key components form the basis of the QGPO algorithm: an  **unconditional diffusion model-based behavior policy**, an **action-state value function** driven by an energy function, and an **intermediate energy guidance function**.

Three key components form the basis of the QGPO algorithm: an  **unconditional diffusion model-based behavior policy**, an **action-state value function** driven by an energy function, and an **intermediate energy guidance function**.

The training of these three models requires two serial steps: first by using offline data sets to train the **unconditional diffusion model-based behavior policy** until convergence, and then alternately train the **action-state value function** and the **intermediate energy guidance function** until convergence.

Training the **action-state value function** requires the use of a training objective based on the Bellman equation. In order to train the **intermediate energy guidance function**, a method called Contrastive Energy Prediction (CEP) is proposed, which is a contrastive learning objective that focuses on maximizing the mutual information between the energy function and energy guidance across identical state-action pairs.

Quick Facts
-----------
1. QGPO deploys as an **offline** RL algorithm.

2. QGPO works as an **Actor-Critic** RL algorithm.

3. QGPO's **Actor** is an energy-based conditional diffusion model constructed on an unconditional diffusion model and an intermediate energy guidance function.

4. QGPO's **Critic** is an action-state value function based on an energy function.

Key Equations or Key Graphs
---------------------------
Using Kullback-Leibler divergence as a constraint to optimize the strategy in reinforcement learning, the optimal strategy :math:`\pi^*` satisfies:

.. math::
    \begin{aligned}
    \pi^*(a|s) \propto \mu(a|s)e^{\beta Q_{\psi}(s,a)}
    \end{aligned}

Here, :math:`\mu(a|s)` functions as the behavior policy, :math:`Q_{\psi}(s,a)` acts as the action-state value function, and :math:eta` is the inverse temperature.

We can regard it as a Boltzmann distribution over action :math:`a` with the energy function :math:`-Q_{\psi}(s,a)` and temperature :math:`\beta`.

In general terms, by denoting :math:`a` with :math:`x0`, the target distribution is as follows:

.. math::
    \begin{aligned}
    p_0(x_0) \propto q_0(x_0)e^{-\beta \mathcal{E}(x_0)}
    \end{aligned}

This distribution can be modeled by an energy-based conditional diffusion model:

.. math::
    \begin{aligned}
    p_t(x_t) \propto q_t(x_t)e^{-\beta \mathcal{E}_t(x_t)}
    \end{aligned}

In this case, :math:`q_t(x_t)` is the unconditional diffusion model, and :math:`\mathcal{E}_t(x_t)` represents the intermediate energy during the diffusion process.

When inferring from the diffusion model, the energy-based conditional diffusion model's score function can be computed as:

.. math::
    \begin{aligned}
    \nabla_{x_t} \log p_t(x_t) = \nabla_{x_t} \log q_t(x_t) - \beta \nabla_{x_t} \mathcal{E}_t(x_t)
    \end{aligned}

where :math:`\nabla_{x_t} \log q_t(x_t)` works as the unconditional diffusion model's score function, and :math:`\nabla_{x_t} \mathcal{E}_t(x_t)` acts as the score function of intermediate energy, referred to as energy guidance.

.. figure:: images/qgpo_paper_figure1.png
   :align: center

As an energy-based conditional diffusion modeled policy, QGPO comprises three components: a behavior policy based on the unconditional diffusion model, an energy function based action-state value function, and an intermediate energy guidance function.

Thus, the training process of QGPO comprises three stages: unconditional diffusion model training, energy function training, and energy guidance function training.

Initially, the unconditional diffusion model receives training from an offline dataset by minimizing the unconditional diffusion model's negative log-likelihood, which switches into minimizing the weighted MSE loss over score function of the unconditional diffusion model:

.. math::
    \begin{aligned}
    \mathcal{L}_{\theta} = \mathbb{E}_{t,x_0,\epsilon} \left[ \left( \epsilon_{\theta}(x_t,t) - \epsilon \right)^2 \right]
    \end{aligned}

where :math:`\theta` is the parameters of unconditional diffusion model.

In QGPO, the behavior policy over action :math:`a` conditioned by state :math:`s` is defined as the unconditional diffusion model, it can be written as:

.. math::
    \begin{aligned}
    \mathcal{L}_{\theta} = \mathbb{E}_{t,s,a,\epsilon} \left[ \left( \epsilon_{\theta}(a_t,s,t) - \epsilon \right)^2 \right]
    \end{aligned}

where :math:`x_0` is the initial state, :math:`x_t` is the state after :math:`t` steps of diffusion process.

Secondly, the state-action value function can be calculated using an in-support softmax Q-Learning method:

.. math::
    \begin{aligned}
    \mathcal{T}Q_{\psi}(s,a) &= r(s,a) + \gamma \mathbb{E}_{s' \sim p(s'|s,a), a' \sim \pi(a'|s')} \left[ Q_{\psi}(s',a') \right] \\
    &\approx r(s,a) + \gamma \frac{\sum_{\hat{a}}{e^{\beta_Q Q_{\psi}(s',\hat{a})}Q_{\psi}(s',\hat{a})}}{\sum_{\hat{a}}{e^{\beta_Q Q_{\psi}(s',\hat{a})}}}
    \end{aligned}

Here :math:`\psi` refers to the parameters of the action-state value function, and :math:`\hat{a}` is the action sampled from the unconditional diffusion model.

Thirdly, the energy guidance function receives training by minimizing the contrastive energy prediction (CEP) loss:

.. math::
    \begin{aligned}
    \mathcal{L}_{\phi} = \mathbb{E}_{t,s,\epsilon^{1:K},a^{1:K}\sim \mu(a|s)}\left[-\sum_{i=1}^{K}\frac{\exp(\beta Q_{\psi}(a^i,s))}{\sum_{j=1}^{K}\exp(\beta Q_{\psi}(a^j,s))}\log{\frac{\exp(f_{\phi}(a_t^i,s,t))}{\sum_{j=1}^{K}\exp(f_{\phi}(a_t^j,s,t))}}\right]
    \end{aligned}

In this case, :math:`\phi` denotes the parameters of energy guidance function.

After training, the action generation of the QGPO policy is a diffusion model sampling process conditioned on the current state, which combines the output of both unconditional diffusion model-based behavior policy and the gradient of the intermediate energy guidance function.
Its scoring function can be calculated as:

.. math::
    \begin{aligned}
    \nabla_{a_t} \log p_t(a_t|s) = \nabla_{a_t} \log q_t(a_t|s) - \beta \nabla_{a_t} \mathcal{E}_t(a_t,s)
    \end{aligned}

Then use **DPM-Solver** to solve and sample the diffusion model and obtain the optimal action:

.. math::
    \begin{aligned}
    a_0 &= \mathrm{DPMSolver}(\nabla_{a_t} \log p_t(a_t|s), a_1) \\
    a_1 &\sim \mathcal{N}(0, I)
    \end{aligned}

Implementations
----------------
The default config is defined as follows:

.. autoclass:: ding.policy.qgpo.QGPOPolicy

Model
~~~~~~~~~~~~~~~~~
Here we provide examples of `QGPO` model as default model for `QGPO`.

.. autoclass:: ding.model.QGPO
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:


Benchmark
-----------

.. list-table:: Benchmark and comparison of QGPO algorithm
   :widths: 25 15 30 15 15
   :header-rows: 1

   * - environment
     - best mean reward
     - evaluation results
     - config link
     - comparison
   * - | Halfcheetah
       | (Medium Expert)
     - 11226
     - .. image:: images/benchmark/halfcheetah_qgpo.png
     - `config_link_1 <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/halfcheetah_qgpo_medium_expert_config.py>`_
     - | d3rlpy(12124)
   * - | Walker2d
       | (Medium Expert)
     - 5044
     - .. image:: images/benchmark/walker2d_qgpo.png
     - `config_link_2 <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/walker2d_qgpo_medium_expert_config.py>`_
     - | d3rlpy(5108)
   * - | Hopper
       | (Medium Expert)
     - 3823
     - .. image:: images/benchmark/hopper_qgpo.png
     - `config_link_3 <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/hopper_medium_expert_qgpo_config.py>`_
     - | d3rlpy(3690)


**Note**: the D4RL environment used in this benchmark can be found `here <https://github.com/rail-berkeley/d4rl>`_.

References
-----------
- Lu, Cheng, et al. "Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning.", 2023; [https://arxiv.org/abs/2304.12824].

Other Public Implementations
----------------------------

- `Official implementation`_

.. _`Official implementation`: https://github.com/ChenDRAG/CEP-energy-guided-diffusion
