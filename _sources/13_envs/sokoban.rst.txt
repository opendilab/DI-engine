Sokoban
~~~~~~~

Overview
=========

Sokoban is a discrete action space environment. In this game, the agent needs to choose an action from a set of discrete actions, and the goal is to push all the boxes in the room to a specified position.The Sokoban environment here implements the random generation of rooms to avoid deep neural networks training overfitting for predefined rooms.

.. image:: ./images/sokoban.gif
   :align: center
   :scale: 70%

Installation
=============

Method
--------

Users could choose to install via pip or pip local installation after git clone the repository.

Note: If the user does not own root authority, please add ``--user`` after the install command.

.. code:: shell

   # Method1: Install Directly
   pip install gym-sokoban
   # Method2: Install from source code
   git clone git@github.com:mpSchrader/gym-sokoban.git
   cd gym-sokoban
   pip install -e .

Verify Installation
----------------------

After the installation is complete, run the following Python program, if no error is reported, then the installation is successful.

.. code:: python

   import gym
   import gym_sokoban
   env = gym.make('Sokoban-v0')
   obs = env.reset()
   print(obs.shape) # (160, 160, 3)

Space before transformation (Original Environment)
====================================================


Observasion Space
---------------------

-  The actual game screen, RGB three-channel image, the specific size is \ ``(160, 160, 3)`` \ (take Sokoban-v0 as an example), the data type is \ ``uint8`` \

-  Each room contains five main elements: wall, floor, box, target point and players, identified in different colors in this game. The color also changes when the box and the player coincide with the target point.

Action Space
-----------------

-  The game provides 9 kinds of environmental interaction actions, forming a discrete action space of size 9, the data type is \ ``int`` \, and needs to import python values ​​(or a 0-dimensional np array, for example, action 3 is \ ``np.array(3)`` \). Action takes value in 0-8, the specific meaning is:


   -  0: No Operation

   -  1: Push Up

   -  2: Push Down

   -  3: Push Left

   -  4: Push Right

   -  5: Move Up

   -  6: Move Down

   -  7: Move Left

   -  8: Move Right

-  Where Move means move only , the next grid in the corresponding direction needs to have no boxes or walls.

-  Push means to move adjacent boxes, the next grid of the box is required to be free, and two adjacent boxes cannot be pushed directly. If there are no boxes in the adjacent grids in the corresponding direction, Push and Move have the same effect.

Reward Space
--------------

-  The game score is generally a \ ``float`` \value, the specific values are as follows:

   -  Move one step: -0.1

   -  Push the box to the target point: 1.0

   -  Push the box away from the target point: -1.0

   -  Push all boxes to the target point: 10.0


Termination Condition
----------------------

-  All boxes are pushed to the target point, or the number of action steps reaches the maximum number max_step , the current environment episode ends. The default max_step is 120 steps, which can be adjusted in config.

Buit-in Environment
---------------------

-  Sokoban includes 9 environments, \ ``Sokoban-v0`` \,\ ``Sokoban-v1`` \, \ ``Sokoban-v2`` \, \ ``Sokoban-small-v0`` \, \ ``Sokoban-small-v1`` \,\ ``Sokoban-large-v0`` \,\ ``Sokoban-large-v1`` \,\ ``Sokoban-large-v2`` \,\ ``Sokoban-huge-v0`` \.The environments only differs in the size of the room and the number of boxes, and the internal environment of the room is randomly generated.

-  For example, \ ``Sokoban-v0`` \ means the room size is 10*10 and there are 3 boxes in the room. After each reset, the environment will be randomly generated based on the room size and the number of boxes.

-  Since in the random generation process, the box is firstly generated on the target point, and then moves in the opposite direction to the starting point, so all environments have solutions.


Key Facts
==========

1. Sparse reward environment, positive reward is only obtained when the box is pushed to the target point. The reward value range is small, the maximum value is 10+N, where N is the number of boxes. The minimum value is -max_step .


2. Discrete action space


Other
========

Lazy Initialization
--------------------

In order to facilitate parallel operations such as environment vectorization, environment instances generally implement lazy initialization, that is, the \ ``__init__`` \ method does not initialize the real original environment case, but only sets relevant parameters and configuration values. The concrete original environment instance is initialized when the \ ``reset`` \ method is used.

Random Seeds
--------------

-  There are two parts of random seeds in the environment that need to be set, one is the random seed of the original environment, and the other is the random seed of the random library used by various environment transformations (such as \ ``random`` \ , \ ``np.random`` \)

-  For the environment caller, just set these two seeds through the \``seed``\ method of the environment, and do not need to care about the specific implementation details

Concrete implementation inside the environment
----------------------------------------------

-  For the seed of the original environment, set in the \ ``reset`` \ methods of the environment calling function , before the concrete environment implementation  \ ``reset`` \ 

-  For the random library seeds, set the value directly in the \ ``seed`` \ methods of the environment


Store Video
------------

After the environment is created, but before reset, call the \ ``enable_save_replay`` \ method，to specify the path to save the game recording. The environment will automatically save the local video files after each episode ends. (The default implementation is to call \ ``gym.wrappers.RecordVideo`` \ ）, the code shown below will run an environment episode and save the results of this episode in \ ``./video/`` \ ：

.. code:: python

  from easydict import EasyDict
  from dizoo.sokoban.envs.sokoban_env import SokobanEnv

  env = SokobanEnv(EasyDict({'env_id': 'Sokoban-v0', 'is_train': False}))
  env.enable_save_replay('./video')
  obs = env.reset()

  while True:
      action = env.action_space.sample()
      timestep = env.step(action)
      if timestep.done:
          print('Episode is over, eval episode return is: {}'.format(timestep.info['eval_episode_return']))
          break
