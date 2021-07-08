===============================
Tutorial-Developer
===============================

.. toctree::
   :maxdepth: 2

Code Structure
===============

DI-engine
-----------------

.. code:: bash

    ding
    ├── config (Configuration files and utils)
    │   ├── config.py
    │   └── utils.py
    ├── design (Design diagrams)
    ├── docs
    │   ├── Makefile
    │   └── source
    ├── entry (Entries for various pipelines and CLI)
    │   ├── application_entry.py
    │   ├── cli.py
    │   ├── dist_entry.py
    │   ├── parallel_entry.py
    │   ├── predefined_config.py
    │   ├── serial_entry.py
    │   ├── serial_entry_il.py
    │   └── serial_entry_reward_model.py
    ├── envs (Environment and its wrappers & manager)
    │   ├── common
    │   ├── env
    │   ├── env_manager
    │   └── env_wrappers
    ├── hpc_rl (HPC module for DI-engine)
    │   ├── README.md
    │   └── wrapper.py
    ├── interaction (An interactive service framework)
    │   ├── base
    │   ├── config
    │   ├── exception
    │   ├── master
    │   └── slave
    ├── league (League training module)
    │   ├── algorithm.py
    │   ├── base_league.py
    │   ├── one_vs_one_league.py
    │   ├── player.py
    │   ├── shared_payoff.py
    │   └── starcraft_player.py
    ├── model (RL neural network)
    │   ├── common (Common encoders and heads)
    │   ├── template (Template models used in some RL algorithms)
    │   └── wrapper (Model wrapper)
    ├── policy (RL policy)
    │   ├── a2c.py
    │   ├── atoc.py
    │   ├── base_policy.py (Policy base class)
    │   ├── c51.py
    │   ├── collaq.py
    │   ├── coma.py
    │   ├── command_mode_policy_instance.py
    │   ├── common_utils.py
    │   ├── ddpg.py
    │   ├── dqn.py
    │   ├── il.py
    │   ├── impala.py
    │   ├── iqn.py
    │   ├── policy_factory.py
    │   ├── ppg.py
    │   ├── ppo.py
    │   ├── qmix.py
    │   ├── qrdqn.py
    │   ├── r2d2.py
    │   ├── rainbow.py
    │   ├── sac.py
    │   ├── sqn.py
    │   └── td3.py
    ├── reward_model (Reward model module, including IRL, HER, RND)
    │   ├── base_reward_model.py
    │   ├── gail_irl_model.py
    │   ├── her_reward_model.py
    │   ├── pdeil_irl_model.py
    │   ├── pwil_irl_model.py
    │   ├── red_irl_model.py
    │   └── rnd_reward_model.py
    ├── rl_utils (Utils for RL)
    │   ├── a2c.py
    │   ├── adder.py
    │   ├── beta_function.py
    │   ├── coma.py
    │   ├── exploration.py
    │   ├── gae.py
    │   ├── isw.py
    │   ├── ppg.py
    │   ├── ppo.py
    │   ├── td.py
    │   ├── tests
    │   ├── upgo.py
    │   ├── value_rescale.py
    │   └── vtrace.py
    ├── scripts (Command line scripts)
    │   ├── dijob-qbert.yaml
    │   ├── kill.sh
    │   ├── local_parallel.sh
    │   ├── local_serial.sh
    │   ├── slurm_dist.sh
    │   ├── slurm_dist_multi_gpu.sh
    │   └── slurm_parallel.sh
    ├── torch_utils (Utils related to PyTorch)
    │   ├── checkpoint_helper.py
    │   ├── data_helper.py
    │   ├── distribution.py
    │   ├── loss
    │   ├── math_helper.py
    │   ├── metric.py
    │   ├── network
    │   ├── nn_test_helper.py
    │   └── optimizer_helper.py
    ├── utils (Common utils)
    │   ├── autolog
    │   ├── collection_helper.py
    │   ├── compression_helper.py
    │   ├── data
    │   ├── default_helper.py
    │   ├── design_helper.py
    │   ├── fake_linklink.py
    │   ├── file_helper.py
    │   ├── import_helper.py
    │   ├── k8s_helper.py
    │   ├── linklink_dist_helper.py
    │   ├── loader
    │   ├── lock_helper.py
    │   ├── log_helper.py
    │   ├── pytorch_ddp_dist_helper.py
    │   ├── registry.py
    │   ├── registry_factory.py
    │   ├── segment_tree.py
    │   ├── slurm_helper.py
    │   ├── system_helper.py
    │   ├── time_helper.py
    │   └── type_helper.py
    └── worker
        ├── adapter
        │   └── learner_aggregator.py
        ├── collector
        │   ├── base_parallel_collector.py
        │   ├── base_serial_collector.py
        │   ├── base_serial_evaluator.py
        │   ├── comm
        │   ├── episode_serial_collector.py
        │   ├── one_vs_one_collector.py
        │   ├── sample_serial_collector.py
        │   └── zergling_collector.py
        ├── coordinator (Central coordinator)
        │   ├── base_parallel_commander.py
        │   ├── base_serial_commander.py
        │   ├── comm_coordinator.py
        │   ├── coordinator.py
        │   ├── one_vs_one_parallel_commander.py
        │   ├── operator_server.py
        │   ├── resource_manager.py
        │   └── solo_parallel_commander.py
        ├── learner
        │   ├── base_learner.py
        │   ├── comm
        │   └── learner_hook.py
        └── replay_buffer
            ├── advanced_buffer.py
            ├── base_buffer.py
            ├── episode_buffer.py
            ├── naive_buffer.py
            └── utils.py

.. note::

    This file tree omits files like ``__init__.py`` and ``test`` (including CI test files)

DI-zoo
---------------------------------

.. code:: bash

    dizoo
    ├── atari
    ├── box2d
    │   ├── bipedalwalker
    │   └── lunarlander
    ├── classic_control
    │   ├── bitflip
    │   ├── cartpole
    │   └── pendulum
    ├── common
    ├── competitive_rl
    ├── gfootball
    ├── mujoco
    ├── multiagent_particle
    ├── pomdp
    └── smac

.. note::

    Each env may contain folders: ``config`` (Confg file), ``Entry`` (Customized entry), ``envs`` (Env that derived from DI-engine base env)

Data Flow
============================

Each training instance of DI-engine can mainly divides 3 parts——Coordinator, Learner, Collector, and some support Middleware, which is shown as follows: 

.. image:: dataflow.png


And the sequence relation of them can be described in the next image:

.. image:: parallel_main-sequence.png
