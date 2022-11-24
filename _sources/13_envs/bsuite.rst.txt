Bsuite
~~~~~~~

Description
============

``bsuite`` is a collection of carefully-designed experiments that investigate core capabilities of a reinforcement learning (RL) agent with two main objectives:

    1. To collect clear, informative and scalable problems that capture key issues in the design of efficient and general learning algorithms.
    2. To study agent behavior through their performance on these shared benchmarks.

.. figure:: ./images/bsuite.png
   :align: center
   :scale: 70%

   Image taken from: https://github.com/deepmind/bsuite

Here we take *Memory Length* as an example environment to illustrate below. It's designed to test the number of sequential steps an agent can remember a single bit. The underlying environment is based on a stylized `T-maze <https://en.wikipedia.org/wiki/T-maze>`__ problem, parameterized by a length :math:`N \in \mathbb{N}`. 
Each episode lasts N steps with observation :math:`o_t=\left(c_t, t / N\right)` and 
action space :math:`\mathcal{A}=\{-1,+1\}`.

   - At the beginning of the episode the agent is provided a context of +1 or -1, which means :math:`c_1 \sim {Unif}(\mathcal{A})`.
   - At all future timesteps the context is equal to zero and a countdown until the end of the episode, which means :math:`c_t=0` for all :math:`t>2`.
   - At the end of the episode the agent must select the correct action corresponding to the context to reward. The reward :math:`r_t=0` for all :math:`t<N`, and :math:`r_N={Sign}\left(a_N=c_1\right)`


.. figure:: ./images/bsuite_memory_length.png
   :align: center
   :scale: 70%

   Image taken from paper `Behaviour Suite for Reinforcement Learning <https://arxiv.org/abs/1908.03568>`__

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
   env = bsuite.load_from_id('memory_len/0') # this environment configuration is 'memory steps' long
   timestep = env.reset()
   print(timestep)

Original Environment Space
===========================

Observations Space
-------------------

-  The observation of agent is a 3-dimensional vector. Data type is ``float32``. Their specific meaning is as below:

  -  obs[0] shows the current time, ranging from [0, 1]. 
  -  obs[1] shows the query as an integer number between 0 and num of bit at the last step. It's always 0 in memory length experiment because there is only a single bit. (It's useful in memory size experiment.)
  -  obs[2] shows the context of +1 or -1 at the first step. At all future timesteps the context is equal to 0 and a countdown until the end of the episode

Actions Space
---------------

-  The action space is a discrete space of size 2, which is {-1,1}. Data type is ``int``.

Rewards Space
-------------

-  The reward space is a discrete space of size 3, which is a ``float`` value.

  -  If it isn't the last step (t<N), the reward is 0.
  -  If it's the last step and the agent select the correct action, the reward is 1.
  -  If it's the last step andthe agent select a wrong action, the reward is -1.

Others
-------

-  Environments terminate once they have reached their maximum number of steps N.


Key Facts
==========

1. We can change the memory length N to make it gradually more challenging.

2. Discrete actions space.

3. Each environment is designed to test a particular propriety of RL policies, including: generalization, exploration, credit assignment, scaling, noise, memory.


Others
=======

Using bsuite in 'OpenAI Gym' format
------------------------------------

Our implementation uses the bsuite Gym wrapper to make the bsuite codebase run under the OpenAI Gym interface. Hence, ``gym`` needs to be installed to make bsuite work properly.

.. code:: python

   import bsuite
   from bsuite.utils import gym_wrapper
   env = bsuite.load_and_record_to_csv('memory_len/0', results_dir='/path/to/results')
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
   cfg = {'env': 'memory_len/15'}
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


Benchmark algorithm performance
===============================

   - memory_len/15 + R2D2

   .. figure:: ./images/bsuite_momery_len_15_r2d2.png
      :align: center
      :scale: 70%
