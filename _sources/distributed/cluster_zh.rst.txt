以集群方式运行（WIP doge）
===============================

.. toctree::
   :maxdepth: 3

如果我们有多台服务器或者 k8s 集群，如何启动并将他们连接起来是一个比较麻烦的问题，这时借助 ``ditask`` 命令将使网络配置变得更简单。

使用 ``ditask`` 可启用新的主流程下的并行模式，在此模式下你需要通过 ``--package path/to/my/package``, ``--main my_module.main`` 指定你的代码包和主函数。

关于主入口文件可参考 `分布式 <./index_zh.html>`_ 章节中的示例，下面是一个极简示例：

.. code-block:: python

   from ding.framework import Task

   def main():
      with Task() as task:
         task.use(lambda _: print("Hello"))
         task.run(max_step=10)

   if __name__ == "__main__":
      main()
      # Parallel.runner(n_parallel_workers=3)(main)

这样你既可以单独运行这个文件，同时也能利用 ``ditask`` 将它启动起来：

.. code-block:: shell

   $ ditask --package my_module --main my_module.main

``ditask`` 会加载你指定的模块和主函数，并且使用并行模式执行主函数，关于并行模式（ ``Parallel`` 模块）的参数，如果阅读过前面的章节，\
相信你已经很熟悉了。

.. note ::

   * ``--package`` 可以指定一个文件夹或者 zip 文件。
   * ``--main`` 中的路径会分隔为 ``模块`` 和 ``主函数`` 两部分，最后一个 ``.`` 号之后的为主函数。例如：``--main root_module.sub_module.main`` \
     则是调用 ``root_module.sub_module`` 中的 ``main`` 函数，假如不指定 ``--main``，会默认尝试调用 ``package`` 模块加 ``main`` 函数

部署在多台服务器上
-------------------------------

假设你现在有三台服务器，ip 分别为 192.168.0.1-3，我们首先选择一台服务器执行如下命令：

.. code-block:: shell

   $ ditask --package path/to/my/package --main my_module.main --parallel-workers 1 --protocol tcp --ports 50515  --node-ids 0

这表示将在这台服务器上启动一个监听 50515 端口的 worker，并且指定他的 node_id 为 0，关于可用的参数可查看 ``Parallel`` 模块的文档或者以下简单说明：

.. note ::

   * ``--parallel-workers`` 如果大于 1 则会启动多个子进程，并监听在不同的端口上，默认为 1。
   * ``--protocol`` 在多机模式下请使用 tcp，可选 ipc，默认为 tcp。
   * ``--ports`` 端口列表，在 n_parallel_workers 大于 1 时可指定为数组，每个 worker 将按此数组分配端口，默认按 50515 递增。
   * ``--node-ids`` 端口 id，在代码中会根据 node_id 来区分哪些中间件会被执行，如果不指定则默认以本机启动的进程顺序编号，默认按 0 递增。
   * ``--labels`` 标签列表，例如可打上 ``learner``，``evaluator`` 等标签，在主入口文件中就可以根据这些标签决定执行哪些中间件，默认为空。
   * ``--attach-to`` 连接的地址列表，例如 192.168.0.1:50515，可以为多个地址，用逗号分隔，默认为空。
   * ``--address`` 监听的地址，不包含端口，例如 192.168.0.1，默认为本机外网地址。

接下来我们在另两台服务器上分别启动新的进程，并且让它们连接到第一台服务器上：

.. code-block:: shell

   # 在服务器 1
   $ ditask --package my_module --main my_module.main --parallel-workers 1 --protocol tcp --ports 50515  --node-ids 1 --attach-to 192.168.0.1:50515
   # 在服务器 2
   $ ditask --package my_module --main my_module.main --parallel-workers 1 --protocol tcp --ports 50515  --node-ids 2 --attach-to 192.168.0.1:50515

这样我们就得到了三台运行 ``main.py`` 的服务器，并且后面两台都能和第一台保持双向通讯。至于内部的训练逻辑，就和单机下的并行模式一样啦。参考 `分布式 <./index_zh.html>`_

部署在 k8s 服务器上
-------------------------------

如果你有 k8s 集群，部署多机并行任务的过程会变得更简单，我们提供了一个新的命令行工具 xxx (抱 orchestrator 大腿)，可以实现一键将服务分发到多个 pod，并且以你希望的模式连接：

.. code-block:: shell

   $ xxx --workers 3 --package path/to/my/package --main my_module.main --topology star

这个命令会帮助你自动执行上述的三条命令，即实现了和多机部署一模一样的效果，即三个 pod，以星型拓扑方式连接。

此外，如果你希望在 pod 中挂载 gpu，可以增加需要的 gpu 数量，例如：

.. code-block:: shell

   $ xxx --workers 3 --gpus 2 --package path/to/my/package --main my_module.main --topology star

这样就会按顺序给头部的 2 个 pod 挂载 gpu，并在 ``ditask`` 任务中增加 ``gpu`` 标签，在主入口文件中就可以根据标签来决定是否训练等等了。
