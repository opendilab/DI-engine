MBPO
^^^^^^^

Overview
---------
Model-based policy optimization (MBPO) was first proposed in `When to Trust Your Model: Model-Based Policy Optimization <https://arxiv.org/abs/1906.08253>`_. MBPO utilizes short model-generated rollouts and provides a guarantee of monotonic improvement at each step. In details, MBPO trains an ensemble of models to fit the transitions of the true environment and uses it to generate short trajectories starting from real environment states to do policy improvement. For the choice of RL policy, MBPO use SAC as its RL component.

Quick Facts
-------------
1. MBPO is a **model-based** RL algorithm.

2. MBPO uses SAC as its RL policy.

3. MBPO only support **continuous** action spaces.

4. MBPOuses **model-eensemble**.


Key Equations or Key Graphs
---------------------------
The maximum likelihood loss used in model training is:

.. math::

   L(w)=\mathbb{E}\left[log( f^w(\boldsymbol{s}_{t+1}|\boldsymbol{s}_t,\boldsymbol{a}_t))\right]

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


P.S.：

1. The above results are obtained by running the same configuration on five different random seeds (0, 1, 2, 3, 4)

Reference
----------

- Michael Janner, Justin Fu, Marvin Zhang, Sergey Levine: “When to Trust Your Model: Model-Based Policy Optimization”, 2019; arXiv:1906.08253.
