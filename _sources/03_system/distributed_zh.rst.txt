分布式
===============================

.. toctree::
   :maxdepth: 2

使用事件系统
-------------------------------

在开始分布式运行之前，我们先介绍一下 DI-engine 的事件系统，所有远程调用都将使用事件系统。

事件系统是一种异步编程范式，好处是可以将不同逻辑的代码解耦，增加可读性，由于异步执行，资源利用率也会有一定提升。
事件系统也有一些缺点，由于异步调用，会给 debug 增加一些困难，我们会在之后的 `代码规范 <../20_spec/index_zh.html>`_ 中介绍一些避免这类缺陷的方法。

task 对象上面提供了五个与事件系统有关的方法 —— ``emit``, ``on``, ``off``, ``once``, ``wait_for``，我们主要介绍 ``emit`` 和 ``on`` 方法，
其他方法都是由这两个方法衍生的。

.. code-block:: python

    with task.start(async_mode=False, ctx=OnlineRLContext()):
        task.on("greeting", lambda msg: print("Get msg: %s" % msg))
        task.emit("greeting", "Hi")

    # >>> Get msg: Hi

上面是事件系统的简单示例，通过 ``task.on`` 注册一个 ``greeting`` 事件的回调方法，通过 ``task.emit`` 触发事件并发送 ``msg`` 参数，参数是可变的，
只要 ``emit`` 和 ``on`` 的回调函数参数能对应即可。接下来我们将这两行代码拆分到不同的代码片段中（用中间件形式编写），它们也可以正常工作：

.. code-block:: python

    def receiver():
        # 由于 on 是永久回调，所以我们只需要注册一次，如果只希望回调一次，可以使用 once
        task.on("greeting", lambda msg: print("Get msg: %s" % msg))
        def _receiver(ctx):
            pass
        return _receiver

    def sender():
        def _sender(ctx):
            task.emit("greeting", "Hi %s times" % ctx.total_step)
        return _sender

    with task.start(async_mode=False, ctx=OnlineRLContext()):
        task.use(receiver())
        task.use(sender())
        task.run(10)

这段代码将在一个进程内发送和接收十次 greeting 事件，下面我们来看看怎样在不同的进程使用它们。

并行化
-------------------------------

``Parallel`` 模块是 DI-engine 中的并行模块，它将使您的 main 函数在多个进程中运行。
它在内部集成了一个消息中间件，配合 task 的事件系统，您可以无感知的在进程间传递信息。

.. image::
    images/event_system.png
    :width: 600
    :align: center

当使用 ``Parallel`` 执行代码时，task 内部的事件系统将自动将消息路由到相连的进程，
这样您就可以像在本地使用事件系统一样，使用来自其他进程的事件和数据。

.. code-block:: python

    def main():
        with task.start(async_mode=False, ctx=OnlineRLContext()):
            if task.router.node_id == 0:  # 1
                task.on("greeting", lambda msg: print("Get msg: %s" % msg))
            else:
                task.emit("greeting", "Hi")  # 2
            sleep(10)  # 3

    if __name__ == "__main__":
        Parallel.runner(n_parallel_workers=2)(main)  # 4

.. note ::

    1. 您可以通过 ``task.router`` 对象访问 ``Parallel`` 实例，获得当前进程的编号 ``node_id``，以便在不同的进程内执行不同的函数逻辑。
    2. 您可以通过 ``task.emit`` 的 ``only_local`` 和 ``only_remote`` 参数来控制数据的发送对象，默认将广播到所有进程。
    3. 由于 ``task`` 的 ``with statement`` 退出后，将清空所有的注册事件，所以在示例中我们使用 sleep 来防止 task 过早退出。
    4. 您可以通过 api 文档查看 ``Parallel`` 的其他参数，选择更多的网络模式，包括网格连接，星型连接，使用 redis 作为消息中间件等等。

在 Kubernetes 上部署
-------------------------------

我们等价看待分布式运行和多进程运行，所以在代码上不需要做任何修改，即可从单机扩展到多机环境。
但是我们更推荐使用 kubernetes 来部署 DI-engine，以便更好的利用任务管理和资源隔离技术。

为了能在 kubernetes 上一键运行，我们推荐使用 DI-engine 提供的命令行工具 ``ditask`` 来代替直接通过 python 脚本启动程序。
``ditask`` 支持 ``Parallel`` 模块的所有参数，您只需在代码中完成 main 函数，即可通过 ditask 启动程序。

.. code-block:: bash

    ditask --package my_module --main my_module.main --parallel-workers 2

使用 ditask 可以在单机上启动多个进程。当我们使用 kubernetes 启动多个 pod 时，需要将所有 pod 的 ip 信息通过环境变量暴露给每个进程。
为此我们提供了一种特殊的 kubernetes 任务类型 ``DIJob`` 来实现这些环境变量的配置，无需您手动设置。

您可以通过 `DI-orchestrator <https://github.com/opendilab/DI-orchestrator>`_ 安装 DIJob，
安装以后，下面的模板可以帮助您快速在 kubernetes 上部署 DI-engine：

.. code-block:: yaml

    apiVersion: diengine.opendilab.org/v2alpha1
    kind: DIJob
    metadata:
        name: test
    spec:
        minReplicas: 3
        maxReplicas: 3
        template:
            spec:
            containers:
            - name: di-container
              image: opendilab/ding:latest
              imagePullPolicy: Always
              env:
              - name: PYTHONUNBUFFERED
                value: "1"
              resources:
                requests:
                  cpu: 6
                  memory: "10Gi"
                limits:
                  cpu: 6
                  memory: "10Gi"
              command: ["/bin/bash", "-c",]
              args:
              - |
                ditask --package my_module --main my_module.main --parallel-workers 2

.. note::

    以上模板将启动 6 个 DI-engine 进程（3 个 pod，每个 pod 两个进程）
