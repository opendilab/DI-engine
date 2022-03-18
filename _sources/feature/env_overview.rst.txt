Env Overview
===================


BaseEnv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(ding/envs/env/base_env.py)

Overview:
    The Environment module is one of the important modules in reinforcement learning (RL). In some common RL tasks, such as `atari <https://gym.openai.com/envs/#atari>`_ tasks and `mujoco <https://gym.openai.com/envs/#mujoco>`_ tasks, the agent we train is to explore and learn in such an environment. Generally, to define an environment, you need to start with the input and output of the environment, and fully consider the possible observation space and action space. The `gym` module produced by OpenAI has helped us define most of the commonly used academic environments. `DI-engine` also follows the definition of `gym.Env`, and adds a series of more convenient functions to it, making the call of the environment easier to understand.

Implementation:
    The key concepts of the defination of `gym.Env <https://github.com/openai/gym/blob/master/gym/core.py#L8>`_ are methods including ``step()`` and ``reset()``. According to the given ``observation``, ``step()`` interacts with the environment based on the input ``action``, and return related ``reward``. The environment is reset when we call ``reset()``. Regarding the observation, action, and reward in the environment, their definitions and restrictions are also given, which are ``observation_space`` ``action_space`` and ``reward_range``. 

    `ding.envs.BaseEnv` is similar to `gym.Env`, in addition to the above interfaces and features, `ding.envs.BaseEnv` also defines and implements:
    
    1. ``BaseEnvTimestep(namedtuple)``:
    It defines what environment returns when ``step()`` is called, usually including ``obs``, ``act``,  ``reward``,  ``done`` and ``info``. You can inherit from ``BaseEnvTimestep`` and define your owned variables.

    2. space property: Di-engine retains the definitions of ``observation_space`` and ``action_space``, and expands ``reward_range`` to ``reward_space``, making it consistent with the previous two. In the case of a multi-agent environment, ``num_agent`` should also be specified.

    .. note::

        ``shared_memory`` in ``subprocess_env_manager`` depends on ``obs_space``. If you want to use ``shared_memory``, you must make sure use ``observation_space`` is correctly specified.

    3. ``create_collector_env_cfg()``:
    It is used to create config for collectors. It is independent of ``create_evaluator_env_cfg``, which is convenient for users to set different environmental parameters for data collection and performance evaluation. According to the incoming initial configuration, a corresponding configuration file is generated for each specific environment. By default, The number of environments in the configuration file will be obtained, and then copy the corresponding number of copies of the default environment configuration to return.

    4. ``create_evaluator_env_cfg():``
    It is to create a corresponding environment configuration file for performance evaluation. The function is the same as the above description.

    5. ``enable_save_replay()``:
    The environment can save the running process as a video file, which is convenient for debugging and visualization. It is generally called before the actual running of the environment, and it is often used for notifying the env should save replay and the save path. (This method is optional to implement).

    Besides, some changes have also been made to the details in `ding.envs.BaseEnv`, such as:

    1. seed(): For the seed control of various processing functions and wrappers in the environment, the external setting is the seed of the seed. When the environment is used to collect data, it is also common to distinguish between **static torrents and dynamic torrents**

    2. **Lazy init** is used by default, which means, init only sets the parameters, and the environment is initialized and set seed at the first reset.

    3. At the end of the episode, return the ``final_eval_reward`` key-value pair in the ``BaseEnvTimestep.info`` dict 

    .. note::

        For the specific creation of an environment (such as opening other simulator clients), this behavior should not be implemented in the ``__init__`` method, because there are scenarios that create model instances but do not run (such as obtaining information such as the dimensions of the environment observation). It is recommended to implement in the ``reset`` method, which means to determine whether the operating environment has been created, if not, create and then reset, or directly reset the existing environment. If the user still wants to complete this function in the ``__init__`` method, please confirm by yourself that there will be no waste of resources or conflicts.

     .. note::

        Regarding ``BaseEnvTimestep``, if there is no special requirement, you can directly call the default definition provided by DI-engine, namely:

        .. code:: python

            from ding.envs import BaseEnvTimestep

        If you need to customize it, you can use ``namedtuple(BaseEnvTimestep)``.

    .. tip::

        The ``seed()`` method is usually called after the ``__init__`` method and before the ``reset()`` method. If the creation of the model is placed in the ``reset()`` method, the ``seed()`` method only needs to record this value and set the random seed when the ``reset()`` method is executed.

     .. warning::

        DI-engine has some dependencies on the ``BaseEnvTimestep.info`` field returned by the environment. ``BaseEnvTimestep.info`` returns a dict, and some key-value pairs have related dependencies:
        
        1. ``final_eval_reward``: The key value must be included at the end of an episode of the environment (done=True), and the value is of type float, which means that the environment runs an episode performance measurement

        2. ``abnormal``: Each time step of the environment can contain the key value, the key value is not required, it is an optional key value, and the value is of type bool, indicating whether an error occurred during the operation of the environment, and if it is true, the relevant modules of the will process the step (for example, the relevant data will be removed).

    The class inheritance relationship is shown in the following figure:

        .. image:: images/baseenv_class_uml.png
            :align: center
            :scale: 60%
