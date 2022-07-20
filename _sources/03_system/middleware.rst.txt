Middleware
===============================

.. toctree::
    :maxdepth: 2

In most reinforcement learning processes, there is a 'collect-learn' cycle between the environment and \
the agent -- get data from the environment, train the agent, get better data, and so on. \
We will introduce the characteristics of each environment in the `DI-zoo chapter <... /11_dizoo/index_zh.html>`_, \
and here we will focus on implementing the interaction strategy of the agent.

The complex strategy of reinforcement learning dictates that it is difficult to abstract all the entities involved \
in the interaction with objects, and as better strategies and algorithms continue to emerge, there is an endless supply \
of new concepts and objects. So our idea was not to do object abstraction, but to encapsulate only the process, and to ensure \
that the encapsulated code is reusable and replaceable. This gives rise to the concept of middleware, the foundation of the DI-engine.

.. image::
    images/middleware.png
    :width: 600
    :align: center

As you can see above, each middleware (the green part in the picture) can be presumed by its name alone, \
and you only need to select the appropriate method in the DI-engine's middleware library to combine them and \
complete the entire interaction strategy of the agent.

.. code-block:: python

    with task.start(async_mode=False, ctx=OnlineRLContext()):
        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(eps_greedy_handler(cfg))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(data_pusher(cfg, buffer_))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        task.use(CkptSaver(cfg, policy, train_freq=100))
        task.run(max_step=100000)

Once you are familiar with the middleware, you will see that the major schools of reinforcement learning -- \
Onpolicy, Offpolicy, Offline, etc. -- have so many reusable parts of the process. \
With a few simple selection, you can transform the interaction flow of an offpolicy process into an onpolicy process.

.. code-block:: python

    with task.start(async_mode=False, ctx=OnlineRLContext()):
        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(gae_estimator(cfg, policy.collect_mode))
        task.use(multistep_trainer(cfg, policy.learn_mode))
        task.use(CkptSaver(cfg, policy, train_freq=100))
        task.run(max_step=100000)

Context
-------------------------------

Contexts are messengers that pass data between middleware, and different interaction policies determine what type of context they should use.
For example, ``OnlineRLContext`` and ``OfflineRLContext`` are provided in DI-engine.

.. code-block:: python

    class OnlineRLContext(Context):

        def __init__(self, *args, **kwargs) -> None:
            ...
            # common
            self.total_step = 0
            self.env_step = 0
            self.env_episode = 0
            self.train_iter = 0
            self.train_data = None
            ...

            self.keep('env_step', 'env_episode', 'train_iter', 'last_eval_iter')

The ``OnlineRLContext`` holds the data needed for online training, and the task of each middleware is to use this data and submit new data to the context. \
For example, the task of the OffPolicyLearner middleware is to train the model using ctx.train_data and write the training results back to ctx.train_iter.

At the beginning of each loop, the context is replaced by a new instance, which ensures that the middleware only needs to focus on the data flow within \
a single loop, simplifying the logic and reducing the risk of memory leaks. If you need to save some variables to the next loop, such as env_step, train_iter, \
and other values that need to be accumulated, you can set it as a reserved field with the ctx.keep method.

Using task to execute tasks asynchronously
---------------------------------------------

``Task`` is a global object used by DI-engine to manage reinforcement learning interaction tasks. All runtime state is maintained within task, \
and some syntactic sugar is provided to help make the process easier.

Asynchrony is of great benefit in a time-critical training environment. If the data for the next training (CPU intensive work) can be \
collected while the model is being trained (GPU intensive work), the training time can theoretically be halved. \
To implement the asynchrony, one needs to control complex processes and carefully maintain various states. Now, with middleware and tasks, \
it is possible to change only one parameter to achieve asynchrony in each step.

.. code-block:: python

    # Sequential execution
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        ...

    # Asynchronous execution
    with task.start(async_mode=True, ctx=OnlineRLContext()):
        ...

In addition to training and collection, there are many ways to take advantage of asynchrony, such as moving the next batch of data to the GPU \
earlier while training the model, and evaluating the performance of historical models while training the model. \
In practice, you may want to try more to speed up the whole interaction process by asynchronous execution.

.. image::
    images/async.png
    :align: center

Middleware in different stages
-------------------------------

Most of the middleware can correspond to different stages. You can see the correspondence between the \
existing middleware and the stages in the \
following diagram in order to combine the various middleware correctly.

.. image::
    images/pipeline.png
    :align: center
