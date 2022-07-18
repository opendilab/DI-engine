Env Manager Overview
========================


Env Manager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview:
    env manager is an environment manager that can manage multiple environments of the same type with different configurations. The env manager can run multiple envs at the same time,
    obtain information in the environment at the same time, and provide an interface similar to env, which can greatly simplify the code and speed up the operation.
    The currently supported types are single-process serial and multi-process parallel modes. BaseEnvManager maintains multiple environment instances through cyclic serial (pseudo-parallel),
    and Async(Sync)SubprocessEnvManager uses subprocess vectorization, that is, call multiprocessing, by running env in a child process, manages and runs the environment by means of inter-process communication.
    DI-engine's env manager needs to use the env definition in DI-engine format (or Gym env decorated by EnvWrapper),
    It needs to provide the instantiation interface of each env when it is initialized, and set the specific operation details through config.
    
    Generally speaking, :class:`BaseEnvManager <ding.envs.BaseEnvManager>` is used to run in some simple environments or to debug, and it is recommended to run
    :class:`SyncSubProcessEnvManager <ding.envs.SyncSubProcessEnvManager>` and :class:`AsyncSubProcessEnvManager <ding.envs.AsyncSubProcessEnvManager>`
    in complex environments or a large number of environments for acceleration.

    If you don’t know enough about the env module yet, it is recommended to consult DI-engine's `Env Overview <./env_overview.html>`_

Usage:
    - init
        The initialization of the env manager needs to pass in the instantiation call interface of each env and the config dictionary.
        The lambda function or partial function ``functools.partial`` can be used to wrap the instantiation function of env and specify its operating parameters.


        .. code:: python

            config = dict(
                env=dict(
                    manager=dict(...),
                    ...
                ),
                ...
            )

            # lambda function way
            env_fn = lambda : DI-engineEnv(*args, **kwargs)
            env_manager = BaseEnvManager(env_fn=[env_fn for _ in range(4)], cfg=config.env.manager)

            # partial function way
            from functools import partial
            
            def env_fn(*args, **kwargs):
                return DI-engineEnv(*args, **kwargs)
            env_manager = BaseEnvManager(env_fn=[partial(env_fn, *args, **kwargs) for _ in range(4)], cfg=config.env.manager)

    - launch/reset
        After the env manager's initialization, each environment will not be instantiated immediately. At this time, the env manager will be marked as a `closed` state.
        To initialize the environment for the first time, you need to call the ``launch`` method, which will construct each env instance according to the incoming env instantiation call interface
        (for SubprocessEnvManager, it is to run the subprocess of each environment and establish a communication channel), construct Some of the state variables of the environment are running,
        and the ``reset`` method of each sub-environment is called at the same time to run the environment.
        
        .. warning::

            Calling the ``step`` and ``reset`` methods of env_manager in the `closed` state will cause an exception.

        After calling the ``launch`` method, you can manually reset the sub-environment by calling the ``reset`` method of the env manager.
        When no parameters are passed in, all sub-environments will be reset by default.
        When the ``reset_param`` parameter is passed in, the sub-environment corresponding to the key in ``reset_param`` will be reset, and its key value will be used as the parameter of the sub-environment ``reset`` method.
        Due to the uncertainty of the time required for each sub-environment reset, the env manager will not return the corresponding observation after the step of the sub-environment ends.
        Instead, it will save the return value at the end of the reset and obtain the current value by calling the ``ready_obs`` property.
        Run the observation of the sub-environment that completes the step or reset method, which can speed up the operating efficiency of the SubprocessEnvManager.
        
        .. note::

            When SubprocessEnvManager needs to reset the sub-environments that are being reset, this method will wait for the last reset of these sub-environments to complete before running this reset.

    - step
        The step method will serially (BaseEnvManager) or parallel (SubprocessEnvManager) call the step method of the sub-environment of the env manager, and return the result of the step, and store the observation in the ``ready_obs`` attribute.
        The parameter passed in this method is an ``actions`` dictionary, the key of which specifies the env_id that needs to run the ``step``, and the key value is the action to be run by the ``step`` of the sub-environment.
        According to different env manager types and config settings, when a certain number of sub-environments return step results, this method will check the running results,
        modify the running status of the sub-environments based on these results, and return the result or throw an exception.

        .. warning::

            When ``actions`` contains the sub-environment id that is running other commands or has completed the episode, it will throw an exception.
    
    - ready_obs
        The ``ready_obs`` attribute returns a dictionary containing the env_id of the environment and the key-value pair of the latest observation returned.
        For SubprocessEnvManager, the environment id returned by the ``ready_obs`` attribute must be a sub-environment that has completed the reset or step method and is waiting for a new command.
        Therefore, it is safe to continue to call the ``reset`` and ``step`` of these sub-environments. ``Method. When all sub-environments that are still running (not running to done) have not completed the ``reset`` and ``step`` methods, calling the ``ready_obs`` property will wait for at least one sub-environment to finish running, and Return its observation.

        When using SubprocessEnvManager, as long as the env_id passed to the step and reset methods is the env_id returned by the ready_obs property, there will be no repeated commands for the sub-environment.
    
    - done
        This attribute will judge the completion of all sub-environments (whether it runs to done), if it is, it returns ``True``, otherwise it returns ``False``.
    
    - close
        Like Gym env's ``close`` method, this method will safely close all sub-environments, destroy the processes created by the sub-environments, and release all resources.
        After calling this method, the env manager will be marked as ``closed``, unless it is ``launch`` again to continue using it.

Examples:
    The following is an example of an env manager running multiple environments.

    .. code:: python

        my_env_manager.launch()

        while not finished:
            obs = my_env_manager.ready_obs
            actions = ... # get actions from policy or else.
            timesteps = my_env_manager.step()
            for env_id, timestep in timesteps.item():
                if timestep.done:
                    # without auto_reset
                    my_env_manager.reset(reset_param={env_id: ...})
                    ...

        my_env_manager.close()

Advanced features:
    - auto_reset
        The env manager of DI-engine will automatically reset by default, that is, when an environment runs to done, it will automatically reset to continue running.
        The parameters of reset are the parameters set for the sub-environment during the last manual reset, unless the number of episodes run is accumulated Reach the episode_num specified in config.
        To turn off this feature, you can specify ``auto_reset=False`` in config
    
    - env state
        In order to facilitate the management of the status of each sub-environment and facilitate debugging, the env manager of DI-engine provides an enumerated type of environment status to grasp the running status of all sub-environments in real time.
        The specific meaning is as follows:

        - VOID: The env manager has been initialized, but the sub-environment has not yet been instantiated
        - INIT: The sub-environment has been instantiated and has not yet been launched or reset
        - RUN: sub-environment reset or step completed, running in progress
        - RESET: sub-environment resetting
        - DONE: sub-environment running to done
        - ERROR: The sub-environment has an exception occurred
        
        The conversion between each state is as shown in the figure:

            .. image:: images/env_state.png

    - max_retry 和 timeout
        In order to prevent some sub-environments from reporting errors temporarily due to connection problems, or the program will not exit normally when the sub-processes are stuck, the env manager of DI-engine has added retry protection and timeout detection mechanisms.
        The user can specify the maximum number of retry and the maximum waiting time for communication between reset, step and sub-processes in config. When the waiting time is exceeded, an exception will be thrown in order to terminate the operation early.
        The settings and default values of these parameters in config are as follows:
        
        .. code-block:: python

            manager_config = dict(
                max_retry=1, # max retry times for step and reset, default to 1
                reset_timeout=60, # max waiting time for reset, default to 60s
                retry_waiting_time=0.1, # retry interval time for reset, default to 0.1s
                step_timeout=60, # max waiting time for rstep, default to 60s
                step_wait_timeout=0.01, # retry interval time for step, default to 0.1s
                connect_timeout=60, # max waiting time for communication between child processes, default to 60s
            )

    - difference between Sync ans Async SubprocessEnvManager
        Pending
  
    - shared_memory
        shared_memory can speed up the transfer of large vector data returned by the environment. When the size of variables such as obs returned by the environment exceeds 100kB, it is recommended to set it to True.
        When using shared_memory, you need to use BaseEnvInfo and EnvElementInfo template in the environment info function to specify the dtype corresponding to the shape and value of obs, act, and rew.
  
    - get_attribute
        Pending


BaseEnvManager (ding/envs/env_manager/base_env_manager.py)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview:
    Run multiple environment managers in a cyclic serial manner.

Interfaces:
    1. __init__: Initialization
    2. launch: Initialize all sub-environments and resources required for state management of sub-environments
    3. reset: Reset all environments by default. When reset_param passed in, the sub-environment specified by env_id will be reset. It returns all running results
    4. step: Executes the input action and run a time step. Like reset, you can pass an action dict to operate on certain environments. It returns all running results
    5. seed: Set the environment random seed, you can pass an env_id list to set specific seeds for certain environments
    6. close: Close all environments, release resources

Properties:
    1. env_num: The number of sub-environments
    2. active_env: List of all unfinished environments
    3. ready_obs: Return all the env_id that are not running with the latest observation
    4. done: Whether all the environments have been completed

SubprocessEnvManager (ding/envs/env_manager/subprocess_env_manager.py)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview:
    Inherit BaseEnvManager, create subprocess for each environment using multiprocessing to run multiple environments in paralle.

Interfaces:
    Only the methods that are different or new from BaseEnvManager are listed below

    1. launch: Initialize the process of running each sub-environment, and initialize the resources required for state management of the sub-environment
    2. reset: Send reset command to environmental processes. When reset_param passed in, the reset command is sent to the subprocess specified by env_id. It returns after sending.
    3. step: Send action commands to environmental processes. Like reset, you can pass an action dict to operate on certain environments. It returns all running results.
    4. close: Destroy all sub-process, release resources

Properties:
    Only the attributes that are different or new from BaseEnvManager are listed below

    1. ready_obs: Return all the env_id that finish running step and reset with the latest observation. If all environments are running previous command, wait until at least one finish running
    2. active_env: List of all running environments
