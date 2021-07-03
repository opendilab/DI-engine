Env Overview
===================


BaseEnv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(ding/envs/env/base_env.py)

Overview:
    The Environment module is one of the important modules in reinforcement learning (RL). In some common RL tasks, such as `atari <https://gym.openai.com/envs/#atari>`_ tasks and `mujoco <https://gym.openai.com/envs/#mujoco>`_ tasks, the agent we train is to explore and learn in such an environment. Generally, to define an environment, you need to start with the input and output of the environment, and fully consider the possible observation space and action space. The `gym` module produced by OpenAI has helped us define most of the commonly used academic environments. `DI-engine` also follows the definition of `gym.Env`, and adds a series of more convenient functions to it, making the call of the environment easier to understand.

Implementation:
    The key concepts of the defination of `gym.Env <https://github.com/openai/gym/blob/master/gym/core.py#L8>`_ are methods including ``step()`` and ``reset()``. According to the given ``observation``, ``step()`` interacts with the environment based on the input ``action``, and return related ``reward``. The environment is reset when we call ``reset()``. Inherited from `gym.Env`，`ding.envs.BaseEnv` also implements:

    1. ``BaseEnvTimestep(namedtuple)``:
    It defines what environment returns when ``step()`` is called, usually including ``obs``, ``act``,  ``reward``,  ``done`` and ``info``. You can inherit from ``BaseEnvTimestep`` and define your owned variables.

    2. ``BaseEnvInfo(namedlist)``:
    It defines the basic information of the environment, usually including the number of agents, the dimension of the observation space. DI-engine now implements it with ``agent_num``, ``obs_space``, ``act_space`` and ``rew_space``, and ``xxx_space`` must be inherited from ``EnvElementInfo`` in ``envs/common/env_element.py``. You can add new variables in your ``BaseEnvInfo``.

    .. note::

        ``shared_memory`` in ``subprocess_env_manager`` depends on ``obs_space``. If you want to use ``shared_memory``, you must use ``EnvElementInfo`` or its subclass.

    3. ``info()``:
    It is used to check the shape in environment quickly

    .. note::

        ``info()`` will change ``obs_shape`` / ``act_shape`` / ``rew_shape`` according to the used wrappers. If you want to add new env wrapper, you need to override static method ``new_shape(obs_shape, act_shape, rew_shape)`` and return new shape.

    4. ``create_collector_env_cfg()``:
    It is used to create config for collectors. It is independent of ``create_evaluator_env_cfg``, which is convenient for users to set different environmental parameters for data collection and performance evaluation. According to the incoming initial configuration, a corresponding configuration file is generated for each specific environment. By default, The number of environments in the configuration file will be obtained, and then copy the corresponding number of copies of the default environment configuration to return.

    5. ``create_evaluator_env_cfg():``
    It is to create a corresponding environment configuration file for performance evaluation. The function is the same as the above description.

    6. ``enable_save_replay()``:
    The environment can save the running process as a video file, which is convenient for debugging and visualization. It is generally called before the actual running of the environment, and functionally replaces the render method in the common environment. (This method is optional to implement).

    Besides, some changes have also been made to the details in `ding.envs.BaseEnv`, such as:

    1. seed(): For the seed control of various processing functions and wrappers in the environment, the external setting is the seed of the seed
    2. Lazy init is used by default, which means, init only sets the parameters, and the environment sets seed at the first reset.
    3. At the end of the episode, return the final_eval_reward key-value pair in the info dict 

    .. note::

        For the specific creation of an environment (such as opening other simulator clients), this behavior should not be implemented in the ``__init__`` method, because there are  scenarios that create model instances but do not run (such as obtaining information such as the dimensions of the environment observation) ). It is recommended to implement in the ``reset`` method, which means to determine whether the operating environment has been created, if not, create and then reset, or directly reset the existing environment. If the user still wants to complete this function in the ``__init__`` method, please confirm by yourself that there will be no waste of resources or conflicts.

     .. note::

        Regarding BaseEnvInfo and BaseEnvTimestep, if there is no special requirement, you can directly call the default definition provided by DI-engine, namely:

        .. code:: python

            from ding.envs import BaseEnvTimestep, BaseEnvInfo

        If you need to customize it, use ``namedtuple(BaseEnvTimestep)`` / ``namedlist(BaseEnvInfo)`` to achieve it in accordance with the above requirements.

    .. tip::

        The ``seed()`` method is usually called after the ``__init__`` method and before the ``reset()`` method. If the creation of the model is placed in the ``reset()`` method, the ``seed()`` method only needs to record this value and set the random seed when the ``reset()`` method is executed.

     .. warning::

        DI-engine has some dependencies on the ``info()`` field returned by the environment. ``info()`` returns a dict, and some key-value pairs have related dependencies:
        
        1. ``final_eval_reward``: The key value must be included at the end of an episode of the environment (done=True), and the value is of type float, which means that the environment runs an episode performance measurement

        2. ``abnormal``: Each time step of the environment can contain the key value, the key value is not required, it is an optional key value, and the value is of type bool, indicating whether an error occurred during the operation of the environment, and if it is true, the relevant modules of the will process the step (for example, the relevant data will be removed).

    The class inheritance relationship is shown in the following figure:

        .. image:: images/baseenv_code.jpg
            :align: center

EnvElement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(ding/envs/common/env_element.py)

Overview:
    EnvElement is the base class of environment elements. ``Observation``, ``action``, ``reward``, etc. can be regarded as environmental elements. This class and its subclasses are responsible for the basic information and processing function definitions of a specific environmental element. This class and its subclasses are stateless and maintain static properties and methods.

Variables:
    1. ``info_template``: Environment elements information template, generally including dimensions, value conditions, processing functions for data sent to the agent, and processing functions for data received from the agent.
    2. ``_instance``: The class variable used to implement the singleton model, pointing to the only instance of the class.
    3. ``_name``: The unique identification name of the class.

Class interface method:
    1. ``__init__``: Initialization, note that after the initialization is completed, the ``_check()`` method will be called to check whether it is legal.
    2. ``info``: return the basic information and processing function of the element class.
    3. ``__repr__``: Returns a string that provides an element description.

The override methods need to be inherited in subclasses:
    1. ``_init``: The actual initialization method, which is implemented so that the subclass must also call the ``_check`` method when calling the method ``__init__``, which is equivalent to ``__init__`` is just a layer of wrapper.
    2. ``_check``: Check the legitimacy method, check whether an environment element class implements the required attributes, the subclass can extend the method, that is, override the method-call the method of the parent class and implement the part that needs to be checked by itself.
    3. ``_details``: element class details.


EnvElementRunner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(ding/envs/common/env_element_runner.py)

Overview:
    The runtime base class of environment elements is implemented using decoration patterns, responsible for runtime-related state management (such as maintaining some state record variables) and providing possible polymorphic mechanisms (reprocessing the results returned by static processing functions).
    On the basis of the static environment element interface, the ``get()`` and ``reset()`` interfaces have been added. This class manages the corresponding static environment element instance as its own member variable ``_core``.

Class variables:
    None.

Class interface method:
    1. ``info``: derived from the parent class of the interface, call the corresponding method of the static element in actual use.
    2. ``__repr__``: derived from the parent class of the interface, call the corresponding method of the static element in actual use.
    3. ``get``: To get the element value at actual runtime, you need to pass in the specific env object. All access to env information is concentrated in the ``get`` method. It is recommended that the access information be implemented through the env property.
    4. ``reset``: restart state, generally need to be called when env restarts.

The override method need to inherit in subclasses:
    1. ``_init``: The actual initialization method, which is implemented so that the subclass must also call the ``_check`` method when calling the method ``__init__``, which is equivalent to ``__init__`` is just a layer of wrapper.
    2. ``_check``: Check the legitimacy method, check whether an environment element class implements the required attributes, the subclass can extend the method, that is, override the method-call the method of the parent class + implement the part that needs to be checked by itself

.. note::


    1. The two classes of ``EnvElement`` and ``EnvElementRunner`` constitute a complete environment element. The former represents static information (stateless), and the latter is responsible for information that changes at runtime (stateful). It is recommended to state related to specific environmental elements. Variables are always maintained here, and only general state variables are maintained in env.
    2. The simple logic diagram of the environment element part is as follows:

        .. image:: images/env_element_class.png

.. note::

    1. All code implementations to the idea of ​​** questioning external input and being responsible for external output**, do necessary checks on input parameters, and clearly define the format of output (return value).
    2. If the key value of the environment element is empty, always use ``None``.
