Gym-Super-Mario-Bros
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Overview
=========
Here is the "Super Mario Bros" series of games, in which players need to control Mario to move and jump, avoid the pits and enemies in the process of leading to the end, gain more gold coins to get higher scores. This game also has many interesting props to enhance player experiences. `gym-super-mario-bros <https://github.com/Kautenja/gym-super-mario-bros>`_ , this environment is encapsulated from "Super Mario Bros" of Nintendo after OpenAI Gym.
Here is the screeshot of game:

.. image:: ./images/mario.png
   :align: center
   :scale: 70%

Installation
==============

Installation Method
---------------------

.. code:: shell

    pip install gym-super-mario-bros


Verify Installation
---------------------

Run the following Python program, and if no errors are reported, the installation is successful.

.. code:: python 

    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    done = True
    for step in range(5000):
        if done:
            state = env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        env.render()

    env.close()

Solution for Installation Failure
-----------------------------------

A common error:

.. code:: shell

    Traceback (most recent call last):
    File "test_mario.py", line 13, in <module>
        state, reward, done, info = env.step(env.action_space.sample())
    File "/Users/wangzilin/opt/anaconda3/envs/mario_test/lib/python3.8/site-packages/nes_py/wrappers/joypad_space.py", line 74, in step
        return self.env.step(self._action_map[action])
    File "/Users/wangzilin/opt/anaconda3/envs/mario_test/lib/python3.8/site-packages/gym/wrappers/time_limit.py", line 50, in step
        observation, reward, terminated, truncated, info = self.env.step(action)
    ValueError: not enough values to unpack (expected 5, got 4)

Due to the updates of gym-super-mario-bros code base cannot keep up with the updates of gym code base sometimes, while executing `pip install gym-super-mario-bros`, the latest gym would be installed by default.The solution is to downgrade gym.
Here gym-super-mario-bros version is 7.4.0, gym version is 0.26.2. We may choose to downgrade gym version to 0.25.1 to solve problems.

.. code:: shell

    pip install gym==0.25.1

Environment Introduction
==========================

Game Rule
-----------

The simulator has two built-in games, Super Mario Bros. and Super Mario Bros.2 . For detailed gameplay and rules, please refer to the wikipedia link at the end of the text.
For Super Mario Bros, in addition to the 32 level, the game also offers the option to play any individual level with one life, one random level (not currently supported in Super Mario Bros 2.).

.. code:: python  

    # Super Mario Bros. 3 lifes from 1-1 to 8-4
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    # Super Mario Bros 2. 3 lifes from 1-1 to 8-4
    env = gym_super_mario_bros.make('SuperMarioBros2-v0')
    # 1 life 3-2
    env = gym_super_mario_bros.make('SuperMarioBros-3-2-v0')
    # 1 life Random level 1-4 2-4 3-4 4-4 (Game end after death,environment would choose another level to begin a new game randomly.)
    env = gym.make('SuperMarioBrosRandomStages-v0', stages=['1-4', '2-4', '3-4', '4-4'])


Keyboard Interaction
----------------------

When you have a display device for rendering, you can try to operate with the keyboard. The environment provides a command line interface, which starts as follows:

.. code:: shell

    # Start 1-4 level
    gym_super_mario_bros -e 'SuperMarioBrosRandomStages-v0' -m 'human' --stages '1-4'


Action Space
--------------

The action space of gym-super-mario-bros contains the whole 256 discrete actions from Nintendo.
To compress this size (and to facilitate learning by the intelligences), the environment provides the action wrapper ``JoypadSpace`` by default to reduce the action dimension: the optional set of actions and their meanings are as follows:

.. code:: python

    # actions for the simple run right environment
    RIGHT_ONLY = [
        ['NOOP'],
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['right', 'A', 'B'],
    ]


    # actions for very simple movement
    SIMPLE_MOVEMENT = [
        ['NOOP'],
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['right', 'A', 'B'],
        ['A'],
        ['left'],
    ]


    # actions for more complex movement
    COMPLEX_MOVEMENT = [
        ['NOOP'],
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['right', 'A', 'B'],
        ['A'],
        ['left'],
        ['left', 'A'],
        ['left', 'B'],
        ['left', 'A', 'B'],
        ['down'],
        ['up'],
    ]

for instance:

.. code:: python

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    # use SIMPLE_MOVEMENT
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # or set your own action space to choose actions like jump to the left and to the right side
    env = JoypadSpace(env, [["right"], ["right", "A"]])


For the 7-dimensional discrete action space represented by SIMPLE_MOVEMENT, the definition using the gym environment space can be expressed as:

.. code:: python

    action_space = gym.spaces.Discrete(7)

State Space
------------

The state space input to gym-super-mario-bros is the image information, and the tensor matrix in three dimensions (datatype=uint8). In addition, the different versions of the game correspond to the same image resolution ``240*256*3``, but the higher the version, the more abbreviated the image is (pixel blocking), as follows:

.. code:: shell

    >>> # View observation space
    >>> gym_super_mario_bros.make('SuperMarioBros-v3').observation_space
    Box([[[0 0 0]
    [0 0 0]
    [0 0 0]
    ...
    [0 0 0]
    [0 0 0]
    [0 0 0]]], [[[255 255 255]
    [255 255 255]
    [255 255 255]
    ...
    [255 255 255]
    [255 255 255]
    [255 255 255]]], (240, 256, 3), uint8)

The corresponding game screenshots of ``v3`` are as follows:

.. image:: ./images/mario_v3.png
   :align: center
   :scale: 70%

Reward Space
-------------
We hope Mario could more likely to move to the **right side** , and move **faster** to the end successfully, the setting of the reward for each frame consists of three parts as follows:

1. ``v``:represents the difference in Mario's x-coordinate (which can be interpreted as the velocity to the right) between two consecutive frames, with positive and negative.


2. ``c``:represents the time used per frame, simply understood as a negative REVERSE for each frame, is used to push the intelligence to reach the end faster.


3. ``d``:represents penalty for death, giving a high penalty of -15 if Mario dies.


Total reward ``r = v + c + d``

Reward is being clipped to ``(-15,15)``


Termination Conditions
-----------------------
For gym-super-mario-bros ,the termination condition for each episode of the environment is that any of the following conditions are encountered.

- Mario wins
  
- Mario is dead
  
- Countdown ends

Additional information contained in info
------------------------------------------
At each step of interaction with the environment , the environment returns the ``info`` dictionary, which contains information about the coins acquired, the current accumulated score, the time remaining, and Mario's current coordinates. The details are as follows:

.. list-table:: More Information
   :widths: 15 10 35
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - | coins
     - int 
     - The number of collected coins
   * - | flag_get
     - bool
     - True if Mario reached a flag or ax
   * - | life
     - int 
     - The number of lives left, i.e., {3, 2, 1}
   * - | score
     - int 
     - The cumulative in-game score
   * - | stage
     - int 
     - The current stage, i.e., {1, ..., 4}
   * - | status
     - str 
     - Mario's status, i.e., {'small', 'tall', 'fireball'}
   * - | time
     - int 
     - The time left on the clock
   * - | world
     - int 
     - The current world, i.e., {1, ..., 8}
   * - | x_pos 
     - int 
     - Mario's x position in the stage (from the left)
   * - | y_pos 
     - int 
     - Mario's y position in the stage (from the bottom)

Built-in Environment
----------------------
There are several built-in environments, including \ ``"SuperMarioBros-v0"``、 ``"SuperMarioBros-v1"``、 ``"SuperMarioBros-v2"``、``"SuperMarioBros-v3"`` \ for Super Mario Bros. And "\ ``"SuperMarioBros2-v0"``、 ``"SuperMarioBros2-v1"`` \for Super Mario Bros. 2.
In addition, Super Mario Bros. also allows you to select specific levels to break into, such as \ ``"SuperMarioBros-1-1-v0"`` \ .

Video Store
------------
gym.wrappers.RecordVideo class is used to store video:

.. code:: python

    import gym
    import time
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

    video_dir_path = 'mario_videos'
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir_path,
        episode_trigger=lambda episode_id: True,
        name_prefix='mario-video-{}'.format(time.ctime())
    )

    # run 1 episode
    env.reset()
    while True:
        state, reward, done, info = env.step(env.action_space.sample())
        if done or info['time'] < 250:
            break
    print("Your mario video is saved in {}".format(video_dir_path))
    try:
        # There is a problem with the destructor of the environment, so an exception is needed to avoid error reporting
        del env
    except Exception:
        pass



DI-zoo Runnable Code Example
==============================

Offers a complete gym-super-mario-bros environment config, use DQN as baseline. Please run \ ``mario_dqn_main.py`` \ doc under \ ``DI-engine/dizoo/mario`` \ catalogue.

.. code:: python

    from easydict import EasyDict

    mario_dqn_config = dict(
        exp_name='mario_dqn_seed0',
        env=dict(
            collector_env_num=8,
            evaluator_env_num=8,
            n_evaluator_episode=8,
            stop_value=100000,
            replay_path='mario_dqn_seed0/video',
        ),
        policy=dict(
            cuda=True,
            model=dict(
                obs_shape=[4, 84, 84],
                action_shape=2,
                encoder_hidden_size_list=[128, 128, 256],
                dueling=True,
            ),
            nstep=3,
            discount_factor=0.99,
            learn=dict(
                update_per_collect=10,
                batch_size=32,
                learning_rate=0.0001,
                target_update_freq=500,
            ),
            collect=dict(n_sample=96, ),
            eval=dict(evaluator=dict(eval_freq=2000, )),
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
    mario_dqn_config = EasyDict(mario_dqn_config)
    main_config = mario_dqn_config
    mario_dqn_create_config = dict(
        env_manager=dict(type='subprocess'),
        policy=dict(type='dqn'),
    )
    mario_dqn_create_config = EasyDict(mario_dqn_create_config)
    create_config = mario_dqn_create_config
    # you can run `python3 -u mario_dqn_main.py`


Benchmark Algorithm Performance
===================================

-  SuperMarioBros-x-x-v0

   - SuperMarioBros-1-1-v0 + DQN

   .. image:: images/mario_result_1_1.png
     :align: center

   - SuperMarioBros-1-2-v0 + DQN

   .. image:: images/mario_result_1_2.png
     :align: center

   - SuperMarioBros-1-3-v0 + DQN

   .. image:: images/mario_result_1_3.png
     :align: center


References
=====================
- gym-super-mario-bros `source code <https://github.com/Kautenja/gym-super-mario-bros>`__
- Super Mario Bros. `wikipedia-Super Mario Bros. <https://zh.wikipedia.org/wiki/%E8%B6%85%E7%BA%A7%E9%A9%AC%E5%8A%9B%E6%AC%A7%E5%85%84%E5%BC%9F>`__
- Super Mario Bros 2. `wikipedia-Super Mario Bros 2. <https://zh.wikipedia.org/wiki/%E8%B6%85%E7%BA%A7%E9%A9%AC%E5%8A%9B%E6%AC%A7%E5%85%84%E5%BC%9F>`__
