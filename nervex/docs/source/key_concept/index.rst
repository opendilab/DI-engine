Key Concept
===============================

.. toctree::
   :maxdepth: 3


Here we show some key concepts about reinforcement learning training and evaluation pipeline designed by nerveX. One of basic control flow(serial pipeline) can be described as:

.. image::
   images/serial_pipeline.png
   :align: center

In the following sections, nerveX first introduces the key concepts/components seperately, then combines them like building a special "Evolution Graph" to offer diffenent computation patterns(serial, parallel, dist).

Concept
----------
``Environment`` and ``policy`` is the most two important concepts in the total design scheme, which can also be called description modules, in most cases, the users of nerveX only need to pay
attention to these two components.

``Worker`` module, such as ``learner``, ``collector``, and ``buffer``, are execution modules implementing the cooresponding tasks derived from the description modules. These worker
module are general in many RL algorithms, but the users can also override their own components easily, the only restirction is to obey the basic interface definition.

Last but not least, ``config`` is the recommended tool to control and record the whole pipeline.

.. tip::
  Environment and policy are partially extended from the original definition in other RL papers and frameworks.

Environment
~~~~~~~~~~~~~
nerveX environment is a superset of ``gym.Env``, it is compatibled with almost gym env interfaces and offers some optional interfaces, e.g.: dynamic seed, collect/evaluate setting, `Env Link <../feature/env_overview.html>`_

``EnvManager``, usually called Vectorized Environments in other frameworks, aims to implement parallel environment simulation to speed up data collection. Instead of interacting with 1 environment per collect step, it allows collector to interact with N homogeneous environments per step, which means that ``action`` passed to ``env.step`` is a vector with length of N, and the return value of ``env.step`` (obs, reward, done) is the same as it.

For the convenience of **asynchronous reset** and **unifying asynchronous/synchronous step**, nerveX modifies the inferface of env manager like this:

.. code:: python

   # nerveX EnvManager                                                            # pseudo code in the other RL papers
   env.launch()                                                                   # obs = env.reset()
   while True:                                                                    # while True:
       obs = env.ready_obs                                                              
       action = random_policy.forward(obs)                                        #     action = random_policy.forward(obs)
       timestep = env.step(action)                                                #     obs_, reward, done, info = env.step(action)
       # maybe some env_id matching when enable asynchronous
       transition = [obs, action, timestep.obs, timstep.reward, timestep.done]    #     transition = [obs, action, obs_, reward, done]
                                                                                  #     if done:
                                                                                  #         obs[i] = env.reset(i)
       if env.done:                                                               #     if env.done  # collect enough env frames
           break                                                                  #         break

There are three types EnvManager in nerveX now:

  - BaseEnvManager——**local test and validation**
  - SyncSubprocessEnvManager——parallel simulation for **low fluctuation environment**
  - AsyncSubprocessEnvManager——parallel simulation for **high fluctuation environment**

For the subprocess-type env manager, nerveX use shared memory among different worker subprocesses to the save the cost of IPC, and `pyarrow <https://github.com/apache/arrow>`_ will be a reliable alternative in the following version.

.. note::
   If the environment is some kind of client, like SC2 and CARLA, maybe a new env manager based on python thread can be faster.

.. note::
   If there are some pre-defined neural network in environment using GPU, like the feature extractor VAE trained by self-supervised training before RL training, nerveX recommends to utilze parallel executions in each subprocess rather than stack all the data in main process and then forward this netowrk. Moreover, it is not a elegant method, nerveX will try to find some new flexible and general solution.

Besides, for robustness in pratical usage, like IPC error(broken pipe, EOF) and environment runtime error, nerveX also provide a series of **Error Tolerance** tools, e.g.: watchdog and auto-retry.

All the mentioned features, the users can refer to `EnvManager Overview <../feature/env_manager_overview.html>`_ for more details.

Policy
~~~~~~~

Config
~~~~~~~~~

Worker
~~~~~~~~~~~

Entry(optional)
~~~~~~~~~~~~~~~~~

.. tip::
  If you want to know more details about algorithm implementation, framework design and efficiency optimization, we also provide the documation of `Feature <../feature/index.html>`_, 

Computation Pattern
----------------------

Serial Pipeline
~~~~~~~~~~~~~~~~~

Parallel/Dist Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~
TBD
