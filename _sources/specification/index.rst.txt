Middleware code specification
=========================================

File structure and naming
--------------------------

The middleware in DI-engine can be divided into two categories, one we call ``function``, which is an atomic operation method,
focusing on doing one thing with a few lines of code, such as ``train`` middleware Execute the training of the model;
The other, which we call ``module``, may combine multiple ``functions`` to perform more complex logic.
This classification refers to `pytorch's nn and nn.functional <https://pytorch.org/docs/stable/nn.functional.html>`_.

Essentially they all belong to middleware, and the usage is exactly the same.

In the directory structure, ``module`` is placed directly in the ``middleware`` directory, named after a noun;
``function`` is placed in the ``middleware/functional`` directory, named after a verb or noun.

Multiple middleware of the same type can be written in one file.

.. code-block::

  ding/
    framework/
      middleware/
        functional/collect.py # Function
        collector.py # Module

Class, Function, Parameter
----------------------------

When writing ``function``, functional-style code is recommended due to the brevity of the code; when writing ``module``, classes are recommended. E.g:

.. code-block::python

    # Function writing
    def train(model: Model):
        def _train(ctx: Context):
            ...
        return _train

    # Module writing
    class Trainer:
        def __init__(self, model: Model):
            self._model = model

        def __call__(self, ctx: Context):
            ...

Passing explicitly named arguments is recommended for all functions, passing dict as arguments is deprecated.

Construction method
---------------------

Most middleware has two layers of methods, such as the outer function of ``function`` and the ``__init__`` function of ``module``,
which are used to pass parameters and objects necessary for the middleware to run.

The return function of ``function`` and the ``__call__`` method of ``module`` are the processes that are called cyclically at runtime,
and only support ``ctx`` as one parameter.

It is recommended to instantiate the object externally to pass to the middleware, rather than inside the middleware, to ensure stateless and procedural middleware:

.. code-block::python

    # correct
    def train(model: Model):
        def _train(ctx: Context):
            ...
        return _train

    model = Model()
    train(model)

    # mistake
    def train():
        model = Model()
        def _train(ctx: Context):
            ...
        return _train

    train()

Runtime method
---------------

When writing the return function of ``function`` or the ``__call__`` method of ``module``, there are a few things to keep in mind:

1. If there is an infinite loop in the method, make sure to check the ``task.finish`` condition to exit:

.. code-block::python

    def runtime(ctx: Context):
        while True:
            if task.finish: # Make sure to judge task.finish
            break
        sleep(1)

2. ``task`` supports two modes of sequential execution and asynchronous execution. The data passed by ``ctx`` may not be generated at the same time in the two modes.
    It is necessary to pay attention to judgment in the middleware, and it is best to support both Two modes:

.. code-block::python

    def runtime(ctx: Context):
        if ctx.get("next_obs"): # In asynchronous mode, the collected data may not be used in this iteration, but will be postponed to the next iteration
            ctx.obs = next_obs
            ctx.next_obs = get_obs()
        else: # In synchronous mode, collect data directly and use it for the next trainer
            ctx.obs = get_obs()

3. It is not recommended to open multiple processes inside the middleware, so as to avoid unforeseen problems caused by too many instantiated objects in the front,
   or multi-level nesting of processes. If you need to use multi-process parallelism, you can split the logic into multiple middleware. ,
   using the parallel capabilities of DI-engine to execute:

.. code-block::python

    # correct
    def train1(ctx: Context):
        ...

    def train2(ctx: Context):
        ...

    task.use(train1)
    task.use(train2)

    # mistake
    def train(ctx: Context):
        p1 = mp.Process(target=...)
        p1.start()
        p2 = mp.Process(target=...)
        p2.start()
        p1.join()
        p2.join()

Event naming convention
------------------------

When using the event mechanism in DI-engine, we agree that events are named according to the following specifications:

1. Events for the purpose of passing data, named with ``sending position_data name[_parameter name_parameter value]``,
   for example: league_job_actor_0 (sent from league to actor, passing job data)
2. Events for remote invocation are named with ``receive location_method name``, for example: league_get_job (sent by actor to league, get job)
