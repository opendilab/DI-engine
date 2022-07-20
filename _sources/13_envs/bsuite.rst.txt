Bsuite
~~~~~~~

Description
============

``bsuite`` is a collection of carefully-designed experiments that investigate core capabilities of a reinforcement learning (RL) agent with two main objectives:

    1. To collect clear, informative and scalable problems that capture key issues in the design of efficient and general learning algorithms.
    2. To study agent behavior through their performance on these shared benchmarks.

.. figure:: ./images/bsuite.png
   :align: center

   Imahge taken from: https://github.com/deepmind/bsuite

Installation
=============

How To install
-----------------

You just need to use the command ``pip`` to install bsuite, however it will be automatically installed when installing DI-engine.

.. code:: shell

   # Method1: Install Directly
   pip install bsuite
   # Method2: Install with DI-engine requirements
   cd DI-engine
   pip install ".[common_env]"

Verify Installation
--------------------

Once installed, you can verify whether the installation is successful by running the following command on the Python command line.

.. code:: python

   import bsuite
   env = bsuite.load_from_id('catch/0')
   timestep = env.reset()
   print(timestep)

Original Environment Space
===========================

Observations Space
-------------------

-  Array representing the state of the environment, dimensions and size can vary according to the specific environment. Its datatype is \ ``np.float32``.

Actions Space
---------------

-  The action space is a discrete space of size N which varies according to the environment. This datatype is \ ``int``\ and input is a python integer value（or a np array of dimension 0 such as \ ``np.array(1)``\ to input action 1）.

-  For example, in the Deep Sea environment, N is equal to 2, thus action values ranges from 0 to 1. For their specific meaning, you can refer to the following list:

   -  0：LEFT.

   -  1：RIGHT.

Rewards Space
-------------

-  Rewards are assigned according to the rules of the environments. Rewards are usually a \ ``float``\ value.

Others
-------

-  Environments terminate once they have reached their maximum number of steps or encountered a failure state. All environments have the fixed number of maximum steps, but not all environments have a failure state.

Key Facts
==========

1. Each environment contains several configurations to make it gradually more challenging.

2. Discrete actions space.

3. Each environment is designed to test a particular propriety of RL policies, including: generalization, exploration, credit assignment, scaling, noise, memory.

4. The scale of rewards can vary significantly.

Others
=======

Using bsuite in 'OpenAI Gym' format
------------------------------------

Our implementation uses the bsuite Gym wrapper to make the bsuite codebase run under the OpenAI Gym interface. Hence, ``gym`` needs to be installed to make bsuite work properly.

.. code:: python

   import bsuite
   from bsuite.utils import gym_wrapper
   env = bsuite.load_and_record_to_csv('catch/0', results_dir='/path/to/results')
   gym_env = gym_wrapper.GymFromDMEnv(env)

About Configurations
-----------------------

Configurations are designed to increase the level of difficulty of an environment. For example, in a 5-armed bandit environment, configurations are used to regulate the level of noise to perturb the rewards.
Given a specific environment, all possible configurations can be visualized with the following code snippet.

.. code:: python

   from bsuite import sweep  # this module contains information about all the environments
   for bsuite_id in sweep.BANDIT_NOISE:
   env = bsuite.load_from_id(bsuite_id)
   print('bsuite_id={}, settings={}, num_episodes={}' .format(bsuite_id, sweep.SETTINGS[bsuite_id], env.bsuite_num_episodes))

.. image:: ./images/bsuite_config.png
   :align: center

Using DI-engine, you can create a bsuite environment simply with the name of your desired configuration.

.. code:: python

   from easydict import EasyDict
   from dizoo.bsuite.envs import BSuiteEnv
   cfg = {'env': 'memory_len/0'}
   cfg = EasyDict(cfg)
   memory_len_env = BSuiteEnv(cfg)


DI-zoo Runnable Code
=======================

The full training configuration can be found on `github
link <https://github.com/opendilab/DI-engine/tree/main/dizoo/bsuite/config/serial>`__
. In the following part, we show an example of configuration for the file, ``memory_len_0_dqn_config.py``\, you can run the demo with the following code：

.. code:: python

    from easydict import EasyDict

    memory_len_0_dqn_config = dict(
        exp_name='memory_len_0_dqn',
        env=dict(
            collector_env_num=8,
            evaluator_env_num=1,
            n_evaluator_episode=10,
            env_id='memory_len/0',
            stop_value=1.,
        ),
        policy=dict(
            load_path='',
            cuda=True,
            model=dict(
                obs_shape=3,
                action_shape=2,
                encoder_hidden_size_list=[128, 128, 64],
                dueling=True,
            ),
            nstep=1,
            discount_factor=0.97,
            learn=dict(
                batch_size=64,
                learning_rate=0.001,
            ),
            collect=dict(n_sample=8),
            eval=dict(evaluator=dict(eval_freq=20, )),
            other=dict(
                eps=dict(
                    type='exp',
                    start=0.95,
                    end=0.1,
                    decay=10000,
                ),
                replay_buffer=dict(replay_buffer_size=20000, ),
            ),
        ),
    )
    memory_len_0_dqn_config = EasyDict(memory_len_0_dqn_config)
    main_config = memory_len_0_dqn_config
    memory_len_0_dqn_create_config = dict(
        env=dict(
            type='bsuite',
            import_names=['dizoo.bsuite.envs.bsuite_env'],
        ),
        env_manager=dict(type='base'),
        policy=dict(type='dqn'),
    )
    memory_len_0_dqn_create_config = EasyDict(memory_len_0_dqn_create_config)
    create_config = memory_len_0_dqn_create_config

    if __name__ == '__main__':
        from ding.entry import serial_pipeline
        serial_pipeline((main_config, create_config), seed=0)
