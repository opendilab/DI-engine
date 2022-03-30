中间件（middleware）编写规范
=====================================

文件目录与命名
--------------

DI-engine 内的中间件可分为两类，一种我们称之为 ``function``，是原子化的操作方法，用几行代码专注做一件事，例如 ``train`` 中间件执行模型的训练；\
另一种我们称之为 ``module``，可能组合了多种 ``function``，执行更复杂的逻辑。这种分类方式参考了 `pytorch 的 nn 和 nn.functional <https://pytorch.org/docs/stable/nn.functional.html>`_ 。

本质上它们都属于中间件，用法完全一致。

目录结构上， ``module`` 直接放置在 ``middleware`` 目录中，以名词命名； ``function`` 放置在 ``middleware/functional`` 目录中，以动词或名词命名。

同种类型的多个中间件，可写在一个文件中。

.. code-block::

  ding/
    framework/
      middleware/
        functional/collect.py  # Function
        collector.py  # Module

类，函数，参数
--------------

编写 ``function`` 时，由于代码简短，建议使用函数式风格代码；编写 ``module`` 时，建议使用类。例如：

.. code-block:: python

    # Function 写法
    def train(model: Model):
        def _train(ctx: Context):
            ...
        return _train

    # Module 写法
    class Trainer:
        def __init__(self, model: Model):
            self._model = model

        def __call__(self, ctx: Context):
            ...

所有函数建议传递明确命名参数，不推荐使用 dict 作为参数传递。

构造方法
------------

大部分中间件都有两层方法，例如 ``function`` 的外层函数和 ``module`` 的 ``__init__`` 函数，是为了传递中间件运行时必须的参数和对象。

而 ``function`` 的返回函数和 ``module`` 的 ``__call__`` 方法，则是运行时循环调用的过程，只支持 ``ctx`` 一个参数。

建议在外部实例化对象传递给中间件，而不是在中间件内部实例化，以确保中间件的无状态和过程化：

.. code-block:: python

    # 正确
    def train(model: Model):
        def _train(ctx: Context):
            ...
        return _train

    model = Model()
    train(model)

    # 错误
    def train():
        model = Model()
        def _train(ctx: Context):
            ...
        return _train

    train()

运行时方法
------------

编写 ``function`` 的返回函数或 ``module`` 的 ``__call__`` 方法时，需要注意以下几点：

1. 如果方法中有死循环，确保判断了 ``task.finish`` 条件退出：

.. code-block:: python

    def runtime(ctx: Context):
        while True:
            if task.finish:  # 确保判断 task.finish
                break
            sleep(1)

2. ``task`` 支持顺序执行和异步执行两种模式， ``ctx`` 传递的数据在两种模式下产生的时间不一定相同，在中间件中需要注意判断，并最好同时支持两种模式：

.. code-block:: python

    def runtime(ctx: Context):
        if ctx.get("next_obs"):  # 在异步模式下，采集到的数据可能不在这轮迭代中使用，而会推迟到下一个迭代
            ctx.obs = next_obs
            ctx.next_obs = get_obs()
        else:  # 同步模式下，直接采集数据，给接着的 trainer 使用
            ctx.obs = get_obs()

3. 中间件内部不建议再开多进程，以免因为前面过多的实例化对象，或者进程嵌套多层导致难以预料的问题，如果需要利用多进程并行，可以将逻辑拆分为多个中间件，利用 DI-engine 的并行能力执行：

.. code-block:: python

    # 正确
    def train1(ctx: Context):
        ...

    def train2(ctx: Context):
        ...

    task.use(train1)
    task.use(train2)

    # 错误
    def train(ctx: Context):
        p1 = mp.Process(target=...)
        p1.start()
        p2 = mp.Process(target=...)
        p2.start()
        p1.join()
        p2.join()


事件命名规范
------------

在使用 DI-engine 内的事件机制时，我们约定事件按以下规范命名：

1. 以传递数据为目的的事件，使用 ``发出位置_数据名[_参数名_参数值]`` 命名，例如：league_job_actor_0（由 league 发往 actor，传递 job 数据）
2. 以远程调用为目的的事件，使用 ``接收位置_方法名`` 命名，例如：league_get_job（由 actor 发往 league，获取 job）
