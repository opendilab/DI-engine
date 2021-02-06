.. _header-n0:

Interaction Overview
====================

.. _header-n2:

概述
----

Interaction为一个独立于任何具体业务的交互式服务框架，用于支持简单非阻塞单线程任务。在nerveX中为分布式环境下的多机多卡训练提供支持。

Interaction分为两个端，是Master端和Slave端，分别为任务下达端和任务执行端。

.. _header-n9:

Slave端
-------

Slave端的任务是从Master端接收任务信息，通过重写\ ``_process_task``\ 方法来执行任务，并通过返回值以及异常抛出来进行消息回传。

一个简单的例子，可以通过如下的方式来构建属于自己的\ ``Slave``\ 类：

.. code:: python

   class MySlave(Slave):  # build a slave class
       def _process_task(self, task: Mapping[str, Any]):
           if 'a' in task.keys() and 'b' in task.keys():
               return {'sum': task['a'] + task['b']}
           else:
               raise TaskFail(result={'message': 'ab not found'}, message='A or B not found in task data.')

这个类的业务逻辑为，如果\ ``a``\ 和\ ``b``\ 参数全部存在，则以json格式返回两者的和，不然通过抛出\ ``TaskFail``\ 异常来表示任务执行失败。

而对于Slave类的调用，有两种方式。首先是通过常规方式启动和关闭slave端：

.. code:: python

   slave = MySlave('0.0.0.0', 8080, channel=233)  # instantiate new slave instance
   slave.start()  # start the slave end

   # do something here

   slave.shutdown()  # shutdown the slave (ATTENTION: slave will not be stopped immediately after this)
   slave.join()  # wait until the slave completely stopped

   print("slave has been stopped")

或者通过\ ``with``\ 进行快速调用

.. code:: python

   with MySlave('0.0.0.0', 8080, channel=233):  # start the slave
       run_until_ctrl_c()  # just block here

   print("slave has been stopped")  # after quit the with block, all the resourced will be automatically released, and wait until slave completely stopped

通过上述调用后，在\ ``master``\ 端连接\ :literal:`\`slave`\ 的8080端口，选择频道\ ``233``\ 即可完成连接。

.. _header-n57:

Master端
--------

（未完待续）

.. _header-n54:

常见问题
--------

Q：何为非阻塞单线程任务？以及何故作此设计？

A：即为\ **当Master端下达任务时，如果Slave端空闲，则执行任务；若果Slave端已经有一个任务正在运行，则拒绝该任务请求**\ 。

与之类似的也有类似几个任务模式，定义如下：

-  非阻塞多线程任务：Slave端设有最大任务数量，当Master下达任务时，如果正在执行的任务已经达到最大数量，则拒绝新任务请求。

-  阻塞单线程任务：当Master端下达任务时，如果Slave端空闲，则执行任务；若果Slave端已经有一个任务正在运行，则将新任务加入任务队列，等待之前的任务完成后再执行。

-  阻塞多线程任务：Slave端设有最大任务数量，当Master下达任务时，如果正在执行的任务已经达到最大数量，则将新任务加入任务队列，等待之前的任务完成后再执行。

考虑到\ **强化学习训练并发计算量大，不宜在节点上分散算力**\ ，且需要方便业务层调度管理的实际需求，故此处设计为非阻塞单线程任务模式。
