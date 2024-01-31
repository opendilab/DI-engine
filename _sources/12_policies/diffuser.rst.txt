Diffuser (Plan Diffuser)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview
---------
In the field of reinforcement learning, if we have a dataset :math:`T` that contains a variety of trajectories, the goal of reinforcement learning is to mine and construct a high-performance agent from :math:`T`, hoping that it can be directly deployed to the environment and achieve good returns. From the perspective of model-free Offline RL, the core of the work focuses on how to more accurately estimate the :math:`Q` 
value of each possible state-action pair, especially those that may query out-of-distribution state-action pairs. From the perspective of model-based Offline RL, the goal often becomes how to better fit the dynamic model of the real environment with this batch of data, and then implement the Online RL algorithm on this converged environmental model. After these two types of algorithms are finished, we usually get an explicit or implicit policy that can return an action that interacts with the environment given any environmental state.

The above methods often do not take advantage of the continuity of trajectories in :math:`T`, but instead divide each trajectory into several state transition segments and treat each such segment as an independent sample point for subsequent training. However, we can look at this dataset
:math:`T` from a completely new perspective, treating the entire trajectory as a sample point, thereby changing our goal to modeling the distribution of the entire trajectory. In the end, we can sample trajectories from the distribution with trajectory optimality as a conditional variable.

In recent years, diffusion models have shined in the generation field. Compared with other generative models, such as VAE and GAN, diffusion models have stronger capabilities to model complex distributions. Therefore, researchers have thought about trying to use diffusion models to model the trajectory distribution in
:math:`T`. Diffusion, as proposed in the study `Planning with Diffusion for Flexible Behavior Synthesis <https://arxiv.org/pdf/2205.09991.pdf>`_, represents a research approach that generates trajectories using a diffusion model.


Quick Facts
-------------
1. Diffusion views offline decisions as a sequence model problem.
2. Diffusion use diffusion model generating trajectory

Key Equations or Key Graphs
---------------------------
In diffusion, trajectories are concatenated as follows in the array:

.. math::
    \begin{align} \tau = \begin{bmatrix}s_0 s_1 ... s_T\\a_0 a_1 ... a_T \end{bmatrix} \nonumber\end{align}

Regarding the time dependency between each transition in the trajectory, Diffusion does not emphasize autoregression or Markovian properties, but makes a more relaxed assumption about temporal locality. Diffusion samples trajectories in the plan by iteratively denoising state-action pairs with variable quantities. 
In a single denoising step, a smaller receptive field constrains the model to infer the denoising result based on adjacent frames in the trajectory. 

.. image:: images/diffuser_sample.png

The original paper uses a model composed of repeated (temporal) convolutional residual blocks to meet these standards. The final structure mainly draws on U-Nets commonly used in image diffusion models, but replaces two-dimensional spatial convolution with one-dimensional temporal convolution.
The loss function of this model is:

.. math::
    \mathcal{L}(\theta) = \mathbb{E}_{i, \epsilon, \tau^0}[||\epsilon - \epsilon_\theta(\tau^i, i)||^2]\nonumber

The algorithm transforms the RL problem into a conditional sampling problem. It utilizes a guiding function to evaluate the value of each sample trajectory at every timestep t.
Ultimately, the algorithm selects the best trajectory as its output. The best trajectory is as follows:

.. math::
    p(\mathcal{O}_t = 1) = exp(r(s_t, a_t))

.. math::
    \begin{align} p(\tau| \mathcal{O}_{1:T} = 1) \propto p(\tau)p(\mathcal{O}_{1:T} = 1|\tau) \nonumber \end{align}

Implementations
----------------
The default config is defined as follows:

    .. autoclass:: ding.policy.plan_diffuser.PDPolicy
        :noindex:

The network interface diffusion used is defined as follows:

    .. autoclass:: ding.model.template.diffusion.GaussianDiffusion
        :members: forward
        :noindex:


Benchmark
-----------
.. list-table:: Benchmark and comparison of DT algorithm
   :widths: 25 15 30 15 15
   :header-rows: 1

   * - environment
     - reward_mean
     - evaluation results
     - config link
     - comparison
   * - | Halfcheetah
       | (Halfcheetah-medium-v2)
     - 44.9
     - .. image:: images/benchmark/pd_half_m_benchmark.png
     - `config_link_1 <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/halfcheetah_medium_pd_config.py>`_
     - 42.8
   * - | Halfcheetah
       | (Halfcheetah-medium-expert-v2)
     - 86.6
     - .. image:: images/benchmark/pd_half_m_e_benchmark.png
     - `config_link_2 <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/halfcheetah_medium_expert_pd_config.py>`_
     - 88.9
   * - | Walker2d
       | (Walker2d-medium-v2)
     - 71
     - .. image:: images/benchmark/pd_walker_m_benchmark.png
     - `config_link_3 <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/walker2d_medium_pd_config.py>`_
     - 79.6
   * - | Walker2d
       | (Walker2d-medium-expert-v2)
     - 108.5
     - .. image:: images/benchmark/pd_walker_m_e_benchmark.png
     - `config_link_4 <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/walker2d_medium_expert_pd_config.py>`_
     - 106.9
   * - | Hopper
       | (Hopper-medium-v2)
     - 58.1
     - .. image:: images/benchmark/pd_hopper_m_benchmark.png
     - `config_link_5 <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/hopper_medium_pd_config.py>`_
     - 74.3
   * - | Hopper
       | (Hopper-medium-expert-v2)
     - 97.2
     - .. image:: images/benchmark/pd_hopper_m_e_benchmark.png
     - `config_link_6 <https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/hopper_medium_expert_pd_config.py>`_
     - 103.3

References
-----------
- Michael Janner, Yilun Du, Joshua B.Tenenbaum, Sergey Levine Planning with Diffusion for Flexible Behavior Synthesis https://arxiv.org/pdf/2205.09991

Other Public Implementations
------------------------------
.. _pymarl: https://github.com/jannerm/diffuser
