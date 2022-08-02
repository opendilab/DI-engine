MBPO
^^^^^^^

Overview
---------
Model-based policy optimization (MBPO) was first proposed in `When to Trust Your Model: Model-Based Policy Optimization <https://arxiv.org/abs/1906.08253>`_. 
MBPO utilizes short model-generated rollouts and provides a guarantee of monotonic improvement at each step. 
In details, MBPO trains an ensemble of models to fit the transitions of the true environment and uses it to generate short trajectories starting from real environment states to do policy improvement. 
For the choice of RL policy, MBPO use SAC as its RL component.

See this repo `awesome-model-based-RL <https://github.com/opendilab/awesome-model-based-RL>`_ for more model-based rl papers.


Quick Facts
-------------
1. MBPO is a **model-based** RL algorithm.

2. MBPO uses SAC as RL policy.

3. MBPO only supports **continuous** action spaces.

4. MBPO uses **model-ensemble**.


Key Equations or Key Graphs
---------------------------

Predictive Model
:::::::::::::::::

MBPO utilizes an ensemble of gaussian neural network, each member of the ensemble is:  

.. math::

  p_\theta(\boldsymbol{s}_{t+1}|\boldsymbol{s}_t,\boldsymbol{a}_t) = N(\mu_\theta(\boldsymbol{s}_t,\boldsymbol{a}_t), \Sigma_\theta(\boldsymbol{s}_t,\boldsymbol{a}_t))

The maximum likelihood loss used in model training is:

.. math::

  L(\theta)=\mathbb{E}\left[log(p_\theta(\boldsymbol{s}_{t+1}|\boldsymbol{s}_t,\boldsymbol{a}_t))\right]


Policy Optimization
::::::::::::::::::::

Policy evaluation step:

.. math::
  Q^\pi(\boldsymbol{s}_t,\boldsymbol{a}_t) = \mathbb{E}_\pi[{\sum}_{t=0}^{\infty}\gamma^t r(\boldsymbol{s}_t,\boldsymbol{a}_t)]

Policy improvement step:

.. math::
  \min J_\pi(\phi, D) = \mathbb{E}_{s_t \sim D}[D_{KL}(\pi \| exp\{Q^\pi - V^\pi\})]

Note: This update guarantees that 
:math:`Q^{\pi_{new}}(\boldsymbol{s}_t,\boldsymbol{a}_t) \geq Q^{\pi_{old}}(\boldsymbol{s}_t,\boldsymbol{a}_t)`,
please check the proof on this Lemma2 in the Appendix B.2 in the original `paper <https://arxiv.org/abs/1801.01290>`_.



Pseudo-code
---------------
.. image:: images/MBPO.png
  :align: center
  :scale: 55%

.. note::
  The initial implementation of MBPO only give the hyper-parameters of applying it on SAC, which does not fit DDPG and TD3 well.


Implementations
----------------
The default config of mbpo model is defined as follows:

.. autoclass:: ding.model.model_based.mbpo.EnsembleDynamicsModel
   :noindex:

The entry MBPO used is defined as follows:

.. autoclass:: ding.entry.serial_entry_mbrl.serial_pipeline_mbrl
   :noindex:


Benchmark
-----------


.. list-table:: Benchmark of MBPO algorithm
   :widths: 25 30 15
   :header-rows: 1

   * - environment
     - evaluation results
     - config link
   * - Hopper
     - .. image:: images/benchmark/sac_mbpo_hopper.png
     - `config_link_p <https://github.com/opendilab/DI-engine/blob/main/dizoo/mujoco/config/hopper_sac_mbpo_default_config.py>`_
   * - Halfcheetah
     - .. image:: images/benchmark/sac_mbpo_halfcheetah.png
     - `config_link_q <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/halfcheetah_sac_mbpo_default_config.py>`_
   * - Walker2d
     - .. image:: images/benchmark/sac_mbpo_walker2d.png
     - `config_link_s <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/walker2d_sac_mbpo_default_config.py>`_


P.S.:

1. The above results are obtained by running the same configuration on three different random seeds (0, 1, 2).


Other Public Implementations
-------------------------------
- `mbrl-lib <https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/algorithms/mbpo.py>`_


Reference
----------

- Michael Janner, Justin Fu, Marvin Zhang, Sergey Levine: “When to Trust Your Model: Model-Based Policy Optimization”, 2019; arXiv:1906.08253.
