分布式
===============================

.. toctree::
   :maxdepth: 3

当我们编写好一个 RL 训练任务时，下一个需要关心的问题就是让它跑的更快。除了依靠算法和编译优化来让代码运行的更快，DI-engine 还设计了一套独特的横向扩展方式，让你的代码可以无缝的扩展到更多的 CPU, GPU 或者多机上面。

Task 对象
-------------------------------

首先假设你已经有了这样一段代码（如果没有请回到 `快速开始 <../quick_start/index_zh.html>`_）：

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


现在我们将介绍新引入的 ``task`` 对象，这是我们用于分布式扩展的基础，请将上述循环中的代码用一个方法封装起来，并放到 ``task`` 中：

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

    [1] 我们在此处使用了闭包函数来创造 ``task`` 需要的执行方法，这是为了让示例更加简单，你也可以使用类或其他任何能创造方法的形式，task 真正需要的只是内部这个以 ctx 为参数的方法。

在 ``task`` 中我们提供了一个 ``use`` 方法，对于熟悉某些 web 框架的开发者来说会觉得非常熟悉，例如在 gin, koa 中这是使用「中间件」的一种方式，我们的本意就是让这些拆分开的方法像真正的中间件一样，\
可重复利用，甚至不仅仅在当前的任务中，你可以将它封装成一个函数库，给其他开发者使用。我们希望这种方式能成为一种对开发者友好的扩展方式，能让更多的人参与到 RL 社区的贡献中。

言归正传，RL 训练中必然包含一个无限重复的循环，里面的代码会被重复的执行，我们将问题简化成每次循环是等价的，这样你只需要关注一次循环中做的事即可。\
我们将一次循环的生命周期分为“采集-训练-评估”等多个阶段，你也可以加入更多的阶段，这些阶段将会组成我们 ``task`` 中的最小可执行单元，即一个中间件。

接下来我们看看上面的 ``training`` 函数，试着将它拆成 ``evaluate``, ``collect``, ``train`` 三个函数：

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

这段代码看起来能运行，但是各个中间件之间的耦合实在是有些麻烦，它让代码既难读又难改，这个时候我们一直没提到的 ``ctx`` 就该出场了

.. image:: images/context.png
    :align: center

``ctx`` 是一个 dict 对象，你可以将任意属性添加到 ``ctx`` 上，它负责在不同的中间件之间传递信息，而使各个中间件可以从具体的对象实例中解耦出来。\
每次循环开始时，``task`` 会重新生成一个空的 ``ctx`` 实例，并在各个中间件之间传递，在一次循环结束时销毁。\
假如需要保留 ``ctx`` 上面的属性到下个循环，例如 ``train_iter`` 每次递增，而不是在循环开始时清零，你得用 ``ctx.keep('train_iter')`` 将它保留下来。

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

魔法时间
-------------------------------

现在一切就绪，让我们进入 ``task`` 的魔法时间！我们可能想知道每个执行单元耗费了多长时间，DI-engine 提供了一个 ``StepTimer`` 装饰器，仅需一行代码，即可在显示每个执行单元的时间

.. code-block:: python

    from ding.framework.wrapper import StepTimer
    with Task() as task:
        task.use_step_wrapper(StepTimer(print_per_step=1))  # Add this line
        task.use(evaluate(evaluator))
        ...

.. image::
    images/step_timer.png
    :align: center

知道了每一步的时间之后，我们就可以尝试通过 **异步或并行** 来优化我们的代码执行效率。比如将 evaluate 的阶段放在子线程中，这样就不会阻塞进程，占用我们宝贵的训练时间。\
但是这往往需要复杂的异步编程逻辑， task 能帮你轻松做到这一点，只需添加 async_mode 参数，所有的中间件都将以异步方式执行，并且在循环结束时做一次同步：

.. code-block:: python

    from ding.framework.wrapper import StepTimer
    with Task(async_mode=True) as task:
        task.use_step_wrapper(StepTimer(print_per_step=1))  # Add this line
        task.use(evaluate(evaluator))
        ...

.. note ::

    我们使用协程来实现代码之间的异步，关于协程的官方实现可参考 `asyncio 文档 <https://docs.python.org/3/library/asyncio.html>`_

异步与并行
-------------------------------

受限于 GIL，有时单进程的异步也不能最大化的利用系统资源，这时候我们就需要考虑将代码放到不同的进程上执行（甚至不同的机器上），在并行（或分布式）方面，DI-engine 的处理方式是一致的。\
这时要介绍一个新的对象 ``Parallel``，它将负责将你的 ``task`` 执行单元分布到不同的进程上去执行：

.. code-block:: python

    from ding.framework import Task, Parallel

    def main():
        ...

    Parallel.runner(n_parallel_workers=3, topology="star")(main)

只需要将上述的 ``main`` 函数传给 ``Parallel`` 就可以实现 ``task`` 执行过程的异步。上例的结果将产生三个进程，每个进程都将执行同样的“采集-训练-评估”过程，并且这三个进程会以 star 型\
拓扑形式连接起来（以第一个进程为中心节点），这种连接将让你接下来的优化工作变得异常简单。

上文 ``evaluate`` 这个过程消费了大量的时间，并且它完全与训练任务无关，让我们将它拆分出去放到别的进程上：

.. code-block:: python

    def evaluate(evaluator, model):
        last_train_iter = -1
        def _evaluate(ctx):
            ctx.setdefault("envstep", -1)  # Avoid attribute not existing
            ctx.setdefault("train_iter", -1)

            ### New code
            if task.router.is_active:
                nonlocal last_train_iter
                while True:
                    if ctx.finish:
                        return
                    if task.parallel_ctx.get("state_dict") and task.parallel_ctx.get("train_iter") > last_train_iter:
                        model.load_state_dict(task.parallel_ctx.state_dict)
                        ctx.train_iter = task.parallel_ctx.train_ter
                        ctx.envstep = task.parallel_ctx.envstep
                        last_train_iter = task.parallel_ctx.get("train_iter")
                        break
                    time.sleep(0.01)
            ###

            if evaluator.should_eval(ctx.train_iter):
                stop, reward = evaluator.eval(None, ctx.train_iter, ctx.envstep)
                if stop:
                    ctx.finish = True  # Write finish state
                    return
        return _evaluate

    def train(task, learner, model, replay_buffer):
        def _train(ctx):
            ctx.setdefault("envstep", -1)
            for i in range(cfg.policy.learn.update_per_collect):
                train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
                if train_data is not None:
                    learner.train(train_data, ctx.envstep)
                    ctx.train_iter = learner.train_iter

                    ### New code
                    if task.router.is_active:
                        ctx.state_dict = model.state_dict()
                    ###

            if ctx.finish:
                learner.save_checkpoint()

        return _train

    def main():
        ...
        # Seperate into different middleware
        with Task() as task:
            task.use(evaluate(task, evaluator, model), filter_labels=["standalone", "node.1"])
            task.use(collect(epsilon_greedy, collector, replay_buffer), filter_labels=["standalone", "node.0"])
            task.use(train(task, learner, model, replay_buffer), filter_labels=["standalone", "node.0"])
            task.run(max_step=max_iteration)

    Parallel.runner(n_parallel_workers=2, topology="star")(main)

以上主要更新了两部分代码：

一部分是在 ``task.use`` 中，加入了 ``filter_labels`` 参数，这是为了在分布式模式下判断哪些中间件要在对应的硬件上执行，\
DI-engine 会默认写入 ``standalone``, ``distributed``, ``async``, 以进程顺序编号的 ``node.*`` 等默认标签，你也可以通过环境变量传入不同的标签来加以区分。\

第二部分是在 ``evaluate`` 和 ``train`` 中分别增加了两段代码，``train`` 中将 ``state_dict`` 写入 ``ctx``，因为在 ``parallel`` 模式中我们会在每次循环结束时\
将 ``ctx`` 以消息的形式发送给其他连接的进程（还记得上面的拓扑形式吗），对方收到的 ``ctx`` 会写入 ``task.parallel_ctx`` 对象，所以在 ``evaluate`` 中\
只要循环检测 ``task.parallel_ctx`` 是否有更新，如果有更新则加载 ``ctx`` 上发送过来的 ``state_dict``，并根据单机同样的 ``should_eval`` 条件进行评估即可。

这样执行的结果就可以完全避免 ``evaluate`` 占用训练的时间，在某些环境上，可极大的加速训练过程。

.. toctree::
   :maxdepth: 1

   gobigger_zh
