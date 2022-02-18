Distributed
===============================

.. toctree::
   :maxdepth: 3

When we have written an RL training task, the next concern may be to make it run faster. \
In addition to relying on algorithms and compilation optimizations to make the code run faster, \
DI-engine designed a unique set of horizontal scaling methods that allow your code to seamlessly scale to more CPUs, GPUs or multiple machines.

Task
-------------------------------

First assume you already have a piece of code like this (if not please go back to `Quick Start <../quick_start/index.html>`_):

.. code-block:: python

    def main():
        # Init instances
        tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
        learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
        collector = SampleCollector(
            cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
        )
        evaluator = BaseSerialEvaluator(
            cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
        )
        replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
        # DQN training loop
        eps_cfg = cfg.policy.other.eps
        epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)
        max_iterations = int(1e8)
        for _ in range(max_iterations):
            if evaluator.should_eval(learner.train_iter):
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                if stop:
                    break
            eps = epsilon_greedy(collector.envstep)
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
            replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
            for i in range(cfg.policy.learn.update_per_collect):
                train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
                if train_data is not None:
                    learner.train(train_data, collector.envstep)

We will now introduce the newly introduced ``task`` object, which is the basis for our use in distributed extensions. \
Wrap the code in the above loop in a method and place it in ``task``:

.. code-block:: python

    from ding.rl_utils import get_epsilon_greedy_fn
    from ding.framework import Task

    def training(learner, collector, evaluator, replay_buffer, epsilon_greedy):  # 1

        def _training(ctx):
            if evaluator.should_eval(learner.train_iter):
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                if stop:
                    ctx.finish = True
                    return
            eps = epsilon_greedy(collector.envstep)
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
            replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
            for i in range(cfg.policy.learn.update_per_collect):
                train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
                if train_data is not None:
                    learner.train(train_data, collector.envstep)

        return _training

    def main():
        # Init instances
        tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
        learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
        collector = SampleCollector(
            cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
        )
        evaluator = BaseSerialEvaluator(
            cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
        )
        replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
        # DQN training loop
        eps_cfg = cfg.policy.other.eps
        epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)
        max_iterations = int(1e8)

        # Use task instead of loop
        with Task() as task:
            task.use(training(learner, collector, evaluator, replay_buffer, epsilon_greedy))
            task.run(max_step=max_iteration)

.. note ::

    [1] We have used closures here to create the methods that ``task`` needs to execute, this is to make the example simpler, \
    you can also use classes or any other form that can create methods, all task really needs is this internal method \
    with ctx as the only argument.

We provide a ``use`` method in task, which is very familiar to developers who are familiar with some web frameworks, \
such as gin, koa, which is a way to use "middleware", and our intention is to make these methods like real middleware. \
reusable, not even just for the task at hand, and you can wrap it into a library for other developers to use. \
We hope that this approach will be a developer-friendly extension that will allow more people to participate in \
contributing to the RL community.

Back to business, RL training necessarily contains an infinitely repeating loop in which the code is executed over and over again. \
We simplify the problem to be equivalent to each loop, so you only need to pay attention to what is done in one loop. \
We divide the life cycle of a cycle into multiple stages such as "collection-training-evaluation". \
You can also add more stages. These stages will form the smallest executable unit in our task, that is, a middleware.

Next, let's take a look at the above ``training`` function and try to split it into three functions: ``evaluate``, ``collect``, and ``train``:

.. code-block:: python

    from ding.rl_utils import get_epsilon_greedy_fn
    from ding.framework import Task

    def evaluate(learner, collector, evaluator):
        def _evaluate(ctx):
            if evaluator.should_eval(learner.train_iter):
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                if stop:
                    ctx.finish = True
                    return
        return _evaluate

    def collect(epsilon_greedy, learner, collector, replay_buffer):
        def _collect(ctx):
            eps = epsilon_greedy(collector.envstep)
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
            replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        return _collect

    def train(learner, collector, replay_buffer):
        def _train(ctx):
            for i in range(cfg.policy.learn.update_per_collect):
                train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
                if train_data is not None:
                    learner.train(train_data, collector.envstep)

        return _train

    def main():
        # Init instances
        tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
        learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
        collector = SampleCollector(
            cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
        )
        evaluator = BaseSerialEvaluator(
            cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
        )
        replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
        # DQN training loop
        eps_cfg = cfg.policy.other.eps
        epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)
        max_iterations = int(1e8)

        # Seperate into different middleware
        with Task() as task:
            task.use(evaluate(learner, collector, evaluator))
            task.use(collect(epsilon_greedy, learner, collector, replay_buffer))
            task.use(train(learner, collector, replay_buffer))
            task.run(max_step=max_iteration)

This code seems to work, but the coupling between the various middleware is really troublesome. \
It makes the code difficult to read and difficult to modify. At this time, ``ctx``, which we haven't mentioned, should come out.

.. image:: images/context.png
    :align: center

``ctx`` is a dict object. You can add any attributes to ``ctx``. It is responsible for transferring information between different middleware, \
so that each middleware can be decoupled from specific object instances. At the beginning of each cycle, \
the ``task`` will regenerate an empty ``ctx`` instance and pass it between the middlewares, and will be destroyed at the end of a cycle. \
If you need to retain the above attributes of ``ctx`` to the next loop, for example, ``train_iter`` increments each time, \
instead of reset it to zero at the beginning of the loop, you have to use ``ctx.keep('train_iter')`` to retain it.

.. code-block:: python

    from ding.rl_utils import get_epsilon_greedy_fn
    from ding.framework import Task

    def evaluate(evaluator):
        def _evaluate(ctx):
            ctx.setdefault("envstep", -1)  # Avoid attribute not existing
            ctx.setdefault("train_iter", -1)
            if evaluator.should_eval(ctx.train_iter):
                stop, reward = evaluator.eval(None, ctx.train_iter, ctx.envstep)
                if stop:
                    ctx.finish = True
                    return
        return _evaluate

    def collect(epsilon_greedy, collector, replay_buffer):
        def _collect(ctx):
            ctx.setdefault("train_iter", -1)
            eps = epsilon_greedy(collector.envstep)
            new_data = collector.collect(train_iter=ctx.train_iter, policy_kwargs={'eps': eps})
            replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
            ctx.envstep = collector.envstep
        return _collect

    def train(learner, replay_buffer):
        def _train(ctx):
            ctx.setdefault("envstep", -1)
            for i in range(cfg.policy.learn.update_per_collect):
                train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
                if train_data is not None:
                    learner.train(train_data, ctx.envstep)
                    ctx.train_iter = learner.train_iter
            if ctx.finish:
                learner.save_checkpoint()

        return _train

    def main():
        # Init instances
        tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
        learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
        collector = SampleCollector(
            cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
        )
        evaluator = BaseSerialEvaluator(
            cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
        )
        replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
        # DQN training loop
        eps_cfg = cfg.policy.other.eps
        epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)
        max_iterations = int(1e8)

        # Seperate into different middleware
        with Task() as task:
            task.use(evaluate(evaluator))
            task.use(collect(epsilon_greedy, collector, replay_buffer))
            task.use(train(learner, replay_buffer))
            task.run(max_step=max_iteration)

Magic time
-------------------------------

Now that everything is ready, let's enter the magic time of ``task``! We may want to know how long each execution unit takes. \
DI-engine provides a ``StepTimer`` decorator, which only needs one line of code to display the time of each execution unit.

.. code-block:: python

    from ding.framework.wrapper import StepTimer
    with Task() as task:
        task.use_step_wrapper(StepTimer(print_per_step=1))  # Add this line
        task.use(evaluate(evaluator))
        ...

.. image::
    images/step_timer.png
    :align: center

After knowing the time of each step, we can try to optimize our code execution efficiency through ``async or parallel``. \
For example, put the ``evaluate`` step in a child thread, so that it will not block the process and take up our precious training time. \
But this often requires complex asynchronous programming logic. ``Task`` can help you do this easily. \
Just add the ``async_mode`` parameter, all middleware will be executed asynchronously, and a synchronization will be done at the end of the loop:

.. code-block:: python

    from ding.framework.wrapper import StepTimer
    with Task(async_mode=True) as task:
        task.use_step_wrapper(StepTimer(print_per_step=1))  # Add this line
        task.use(evaluate(evaluator))
        ...

.. note ::

    We use coroutines to achieve asynchrony between codes. For the official implementation of coroutines, please refer to `asyncio <https://docs.python.org/3/library/asyncio.html>`_

Async and parallel
-------------------------------

Limited by the GIL, sometimes the single-process asynchronous can not maximize the use of system resources. \
At this time, we need to consider placing the code for execution on different processes (even on different machines), \
in terms of parallel (or distributed), DI-engine's processing method is the same. \
At this time, a new object, ``Parallel``, will be introduced, which will be responsible for distributing \
your ``task`` execution units to different processes for execution:

.. code-block:: python

    from ding.framework import Task, Parallel

    def main():
        ...

    Parallel.runner(n_parallel_workers=3, topology="star")(main)

You only need to pass the above ``main`` function to ``Parallel`` to achieve asynchronous task execution.
The result of the above example will produce three processes, each of which will perform the same "collection-training-evaluation" \
process, and these three processes will be connected by ``star`` topology (with the first process as the central node).
This connection will make your next optimization work extremely simple.

The above ``evaluate`` process consumes a lot of time, and it has nothing to do with the training task, \
let us split it out and put it on other processes:

.. code-block:: python

    def evaluate(task, evaluator, model):
        def _evaluate(ctx):
            ctx.setdefault("env_step", -1)  # Avoid attribute not existing
            ctx.setdefault("train_iter", -1)

            ### Wait for new model
            if ctx.train_iter > 0:
                learn_output = task.wait_for("learn_output")[0][0]
                ctx.train_iter, ctx.env_step = learn_output["train_iter"], learn_output["env_step"]
                if not evaluator.should_eval(ctx.train_iter):
                    return
                state_dict = learn_output.get("state_dict")
                if not state_dict:
                    return
                model.load_state_dict(state_dict)
            ###
            if evaluator.should_eval(ctx.train_iter):
                stop, reward = evaluator.eval(None, ctx.train_iter, ctx.env_step)
                if stop:
                    task.finish = True  # Write finish state
        return _evaluate

    def train(task, learner, model, replay_buffer, cfg):
        last_eval_iter = 0

        def _train(ctx):
            ctx.setdefault("envstep", -1)
            for i in range(cfg.policy.learn.update_per_collect):
                train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
                if train_data is not None:
                    learner.train(train_data, ctx.envstep)
                    ctx.train_iter = learner.train_iter

                    ### Broadcast state dict
                    if task.router.is_active:
                        nonlocal last_eval_iter
                        if learner.train_iter - last_eval_iter >= cfg.policy.eval.evaluator.eval_freq:
                            learn_output = {
                                "env_step": ctx.env_step,
                                "train_iter": learner.train_iter,
                                "state_dict": model.state_dict()
                            }
                            task.emit("learn_output", learn_output)
                    ###

        return _train

    def main():
        ...
        # Seperate into different middleware
        with Task() as task:
            if task.match_labels(["node.0"]):
                task.use(collect(epsilon_greedy, collector, replay_buffer))
                task.use(train(task, learner, model, replay_buffer, cfg))
            else:
                task.use(evaluate(task, evaluator, model))

            task.run(max_step=max_iteration)

    Parallel.runner(n_parallel_workers=2, topology="star")(main)

The above mainly updated two parts of the code:

Part of it is to use the ``task.match_labels`` method to determine which middleware to execute on the corresponding process in distributed mode, \
DI-engine will write default labels such as ``standalone``, ``distributed``, ``async``, and ``node.*`` numbered in process order by default. \
You can also pass in different tags through environment variables to distinguish. \

The second part is to add two pieces of code to ``evaluate`` and ``train`` respectively. In ``train``, ``state_dict`` is broadcast to each process with ``task.emit``, \
The evaluate process uses ``task.wait_for`` to wait for the broadcast event to update the model and take the next action.

The result of this execution can completely avoid ``evaluate`` taking up the training time, and in some environments, it can greatly speed up the training process.

.. note ::

    1. It is necessary to briefly introduce the event system here. We have added ``emit``, ``on``, ``once``, ``wait_for`` and other event-related methods to ``task`` , \
    in any middleware, if you want to send data or messages to the outside world, you can use the event system.\
    And we connect the distributed system with the local event system in Parallel, so that events sent from process A can also be heard by process B. \
    This way ensures code independence. A rough schematic is as follows:

.. image:: images/event_system.png
    :align: center

.. toctree::
   :maxdepth: 1

   gobigger
