中间件
===============================

.. toctree::
   :maxdepth: 2

在大部分强化学习流程中，都存在着环境与智能体之间的「探索-利用」循环 —— 从环境中取得数据，训练智能体，取得更好的数据，周而复始。
我们将在后续的 `DI-zoo 章节 <../11_dizoo/index_zh.html>`_ 中详细介绍各个环境的特性，这里将着重实现智能体的交互策略。

强化学习的复杂策略决定了它很难用对象抽象所有参与交互的实体，随着更好的策略和算法不断出现，新的概念和对象无穷无尽，
所以我们的主意是不做对象抽象，而只封装过程，并且力保这些封装后的代码可重用，可替换。这就产生了 DI-engine 的基础概念 —— 中间件。

.. image::
    images/middleware.png
    :width: 600
    :align: center

如上图所示，每个中间件（图中绿色部分）仅靠名字即可推测其用途，您仅需在 DI-engine 的 middleware 库中选择合适的方法，将它们组合起来，就完成了整个智能体的交互策略。

.. code-block:: python

    with task.start(async_mode=False, ctx=OnlineRLContext()):
        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(eps_greedy_handler(cfg))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(data_pusher(cfg, buffer_))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        task.use(CkptSaver(cfg, policy, train_freq=100))
        task.run(max_step=100000)

熟悉了中间件的使用之后，您会发现原来强化学习的几大流派 —— Onpolicy, Offpolicy, Offline 等等居然在流程上会有这么多可复用部分，
通过简单的取舍，您就能将一个 Offpolicy 策略的交互流程改造为 Onpolicy 策略。

.. code-block:: python

    with task.start(async_mode=False, ctx=OnlineRLContext()):
        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(gae_estimator(cfg, policy.collect_mode))
        task.use(multistep_trainer(cfg, policy.learn_mode))
        task.use(CkptSaver(cfg, policy, train_freq=100))
        task.run(max_step=100000)

上下文对象（Context）
-------------------------------

Context 是为中间件之间传递数据的信使，不同的交互策略决定了它们该使用什么类型的 context，
例如 DI-engine 中提供了 ``OnlineRLContext`` 和 ``OfflineRLContext`` 两种 context。

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

``OnlineRLContext`` 上面保存了在线训练所需要的数据，每个中间件的任务就是利用这些数据，并提交新的数据到 context 上面，
例如 OffPolicyLearner 中间件的任务就是利用 ctx.train_data 训练模型，并且将训练结果写回到 ctx.train_iter 上面。

在每个循环开始，context 会初始化为新的实例，这确保中间件只需关注一次循环内的数据流，简化了逻辑，也减少了内存泄漏的风险。
如果您需要保存属性到下一个循环，例如 env_step，train_iter 这类需要累加的数值，可以用 ctx.keep 方法将它设置为保留字段。

使用 task 异步执行任务
-------------------------------

``Task`` 是 DI-engine 用来管理强化学习交互任务的全局对象，所有的运行时状态都在 task 内维护，上面也提供了一些语法糖来帮助流程变得更简单。

在分秒必争的训练环境中，异步带来的好处是显而易见的。如果能在训练模型时（GPU 密集工作）采集下一次训练的数据（CPU 密集工作），理论上可以将训练时间缩短一半。
而要实现这种异步，则需要控制复杂的流程，小心翼翼的维护各种状态。现在借助中间件和 task，只需更改一个参数，即可实现各个环节的异步。

.. code-block:: python

    # 顺序执行
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        ...

    # 异步执行
    with task.start(async_mode=True, ctx=OnlineRLContext()):
        ...

除了训练和采集，有很多环节都可以利用异步的好处，例如在训练模型时，将下一批数据提前搬到 GPU 上；在训练模型的同时评估历史模型的表现。
实践中不妨多尝试一下通过异步执行来加速整个交互流程。

.. image::
    images/async.png
    :align: center
