Procgen
~~~~~~~~~

Overview
=========

Procgen Benchmark is a set of randomly generated environments released by OpenAI using 16 exploits (CoinRun, StarPilot, CaveFlyer, Dodgeball, FruitBot, Chaser
, Miner, Jumper, Leaper, Maze, BigFish, Heist, Climber, Plunder, Ninja and BossFight), the full name of procgen is Procedural Generation, which means procedural generation. For the procgen environment, it can generate games of the same difficulty but using different maps, or games using the same map but different difficulties, which can be used to measure the speed at which the model learns general skills, thereby judging the generalization ability of the algorithm to the environment. The image below shows the Coinrun game in it.

.. image:: ./images/coinrun.gif
   :align: center

The following three pictures represent the different inputs from level1 to level3 in the coinrun environment:

.. image:: ./images/coinrun_level1.png
   :align: center
.. image:: ./images/coinrun_level2.png
   :align: center
.. image:: ./images/coinrun_level3.png
   :align: center

Install
========

Installation Method
--------------------

It can be installed by one-click pip or combined with DI-engine. It only needs to install two libraries, gym and gym[procgen].

.. code:: shell

   # Method1: Install Directly
   pip install gym
   pip install gym[procgen]
   # Method2: Install with DI-engine requirements
   cd DI-engine
   pip install ".[procgen_env]"

Verify Installation
--------------------

After the installation is complete, you can verify that the installation was successful by running the following command on the Python command line:

.. code:: python

   import gym
   env = gym.make('procgen:procgen-maze-v0', start_level=0, num_levels=1)
   # num_levels=0 - The number of unique levels that can be generated. Set to 0 to use unlimited levels.
   # start_level=0 - The lowest seed that will be used to generated levels. 'start_level' and 'num_levels' fully specify the set of possible levels.
   obs = env.reset()
   print(obs.shape)  # (64, 64, 3)

Space Before Transformation (Original Environment)
===================================================

Observation Space
------------------

- The actual game screen, RGB three-channel image, the specific size is\ ``(64, 3, 3)``\ , the data type is\ ``float32``\

Action Space
-------------

- The game operation button space, generally a discrete action space of size N (N varies with the specific sub-environment), the data type is \ ``int``\ , you need to pass in python values (or 0-dimensional np arrays, such as actions 3 is\ ``np.array(3)``\ )


-  For example, in the Coinrun environment, the size of N is 5, that is, the action takes a value from 0 to 4. The specific meaning is:

   -  0：NOOP

   -  1：LEFT

   -  2：RIGHT

   -  3：UP

   -  4：DOWN


Bonus Space
------------

- The game score will vary according to the specific game content. Generally, it is a \ ``float`` \ value. For example, in the Coinrun environment, if you eat coins, you will be rewarded 10.0 points, and there are no other rewards.

Other
------

- The end of the game is the end of the current environment episode. For example, in coinrun, if the agent eats coins or the game time exceeds the maximum allowed game time, the game ends.

Key Facts
==========

1. 2D
RGB three-channel image input, three-dimensional np array, size \ ``(3, 64, 64)`` \ , data type \ ``np.float32`` \ , value  \ ``[0, 255]``\

2. Discrete action space

3. Rewards are sparse. For example, in coinrun, you can only get points if you eat coins.

4. The generalization of the environment. For the same environment, there are different levels. Their input, reward space, and action space are the same, but the difficulty of the game is different.

Transformed Space (RL Environment)
===================================

Observation Space
------------------

- Transform content: resize from \ ``(64,64,3)`` \ to \ ``(3, 64, 64)`` \

- Transformation result: 3D np array with size \ ``(3, 84, 84)`` \ , data type \ ``np.float32`` \ , value \ ``[0, 255]`` \

Action Space
-------------

-  Basically no transformation, it is still a discrete action space of size N, but generally a one-dimensional np array, the size is \ ``(1, )`` \ , the data type is \ ``np.int64``

Bonus Space
------------

-  Basically no transformation

The above space can be expressed as:

.. code:: python

   import gym
   obs_space = gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.float32)
   act_space = gym.spaces.Discrete(5)
   rew_space = gym.spaces.Box(low=0, high=10, shape=(1, ), dtype=np.float32)

Other
------

\ ``info``\returned by the method \ ``step``\ must contain\ ``final_eval_reward``\ key - value pair, representing the evaluation metrics of the entire episode, and the cumulative sum of the rewards for the entire episode in Procgen


Other
======

Lazy Initialization
--------------------

In order to support parallel operations such as environment vectorization, environment instances generally implement lazy initialization, that is, the \ ``__init__`` \ method does not initialize the real original environment instance, but only sets relevant parameters and configuration values. In the first call\ ``reset``\  method initializes the concrete original environment instance.

Random Seed
------------

- There are two parts of the random seed in the environment that need to be set, one is the random seed of the original environment, and the other is the random seed of the random library used by various environment transformations (such as\ ``random``\ ，\ ``np.random``\ )

- For the environment caller, just set these two seeds through the \ ``seed`` \ method of the environment, no need to care about the specific implementation details

- Concrete implementation inside the environment: For the seed of the original environment, set before calling the\ ``reset``\ method of the environment, the concrete original environment\ ``reset``\ 

- Concrete implementation inside the environment: For random library seeds, the value is set directly in the \ ``seed`` \ method of the environment

The Difference between Training and Testing Environments
---------------------------------------------------------

- The training environment uses a dynamic random seed, that is, the random seed of each episode is different, and is generated by a random number generator, but the seed of this random number generator is fixed by the \ ``seed`` \ method of the environment ;The test environment uses a static random seed, that is, the random seed of each episode is the same, specified by the \ ``seed`` \ method.

Store Video
------------

After the environment is created, but before reset, call the \ ``enable_save_replay`` \ method, specifying the path to save the game replay. The environment will automatically save the local video files after each episode ends. (The default call \ ``gym.wrapper.Monitor`` \ implementation, depends on \ ``ffmpeg`` \ ), the code shown below will run an environment episode and save the result of this episode in the form\ ``./video/xxx.mp4``\ in a file like this:

.. code:: python

   from easydict import EasyDict
   from dizoo.procgen.coinrun.envs import CoinRunEnv
   env = CoinRunEnv(EasyDict({'env_id': 'procgen:procgen-coinrun-v0'}))
   env.enable_save_replay(replay_path='./video')
   obs = env.reset()
   while True:
       action = env.random_action()
       timestep = env.step(action)
       if timestep.done:
           print('Episode is over, final eval reward is: {}'.format(timestep.info['final_eval_reward']))
           break

DI-zoo Runnable Code Example
=============================

The full training configuration file is at `github
link <https://github.com/opendilab/DI-engine/tree/main/dizoo/procgen/coinrun/entry>`__
Inside, for specific configuration files, such as \ ``coinrun_dqn_config.py`` \ , use the following demo to run:

.. code:: python

   from easydict import EasyDict
   coinrun_dqn_default_config = dict(
       env=dict(
           collector_env_num=4,
           evaluator_env_num=4,
           n_evaluator_episode=4,
           stop_value=10,
       ),
       policy=dict(
           cuda=False,
           model=dict(
               obs_shape=[3, 64, 64],
               action_shape=5,
               encoder_hidden_size_list=[128, 128, 512],
               dueling=False,
           ),
           discount_factor=0.99,
           learn=dict(
               update_per_collect=20,
               batch_size=32,
               learning_rate=0.0005,
               target_update_freq=500,
           ),
           collect=dict(n_sample=100, ),
           eval=dict(evaluator=dict(eval_freq=5000, )),
           other=dict(
               eps=dict(
                   type='exp',
                   start=1.,
                   end=0.05,
                   decay=250000,
               ),
               replay_buffer=dict(replay_buffer_size=100000, ),
           ),
       ),
   )
   coinrun_dqn_default_config = EasyDict(coinrun_dqn_default_config)
   main_config = coinrun_dqn_default_config
   coinrun_dqn_create_config = dict(
       env=dict(
           type='coinrun',
           import_names=['dizoo.procgen.coinrun.envs.coinrun_env'],
       ),
       env_manager=dict(type='subprocess', ),
       policy=dict(type='dqn'),
   )
   coinrun_dqn_create_config = EasyDict(coinrun_dqn_create_config)
   create_config = coinrun_dqn_create_config
   if __name__ == '__main__':
       from ding.entry import serial_pipeline
       serial_pipeline((main_config, create_config), seed=0)

Benchmark Algorithm Performance
================================


-  Coinrun（Average reward equal to 10 is considered a better Agent）

   - Coinrun + DQN

    .. image:: images/coinrun_dqn.svg
     :align: center

-  Maze（Average reward equal to 10 is considered a better Agent）

   - Maze + DQN

    .. image:: images/maze_dqn.svg
     :align: center
