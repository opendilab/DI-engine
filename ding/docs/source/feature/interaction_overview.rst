.. _header-n0:

Interaction Overview
====================

.. _header-n2:

概述
----

Interaction为一个独立于任何具体业务的交互式服务框架，用于支持简单非阻塞单线程任务。在DI-engine中为分布式环境下的多机多卡训练提供支持。

Interaction分为两个端，是Master端和Slave端，分别为任务下达端和任务执行端。

.. _header-n9:

Slave端
-------

Slave端的任务是从Master端接收任务信息，通过重写\ ``_process_task``\ 方法来执行任务，并通过返回值以及异常抛出来进行消息回传。

一个简单的例子，可以通过如下的方式来构建属于自己的\ ``Slave``\ 类：

.. code:: python

   class MySlave(Slave):  # build a slave class, inheriting from Slave base class
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
       # when this line is reached, all the initialization process has been completed
       run_until_ctrl_c()  # just block here

   print("slave has been stopped")  # after quit the slave with block, all the resources will be automatically released, and wait until slave completely stopped

通过上述调用后，在\ ``master``\ 端连接\ ``slave``\ 的8080端口，选择频道\ ``233``\ 即可完成连接。

值得注意的是：

-  为了方便对任务逻辑的实现，\ **请继承Slave类并实现\ ``_task_process``\ 方法**\ ，不要直接使用Slave类创建Slave端。

-  关于channel，可以视为对不同具体业务的一种标识，其概念类似于无线电频道、网络端口等。\ **建议对于特定的业务设定特定的channel值**\ ，不要直接使用0或缺省等容易碰撞的值，以确保当发生误连的时候可以立刻得到反馈。

.. _header-n57:

Master端
--------

Master端的主要任务是对Slave端进行连接，并通过建立的连接对Slave进行任务的下达和接收。

一个简单的例子，可以通过如下方式来构建属于自己的\ ``Master``\ 类：

.. code:: python

   class MyMaster(Master):  # build a master class, inherit from master base class
       pass

与Slave类似，Master端有两种启动方式。首先是通过常规方式启动和停止Master：

.. code:: python

   master = MyMaster('0.0.0.0', 8088, channel=233)  # instantiate new master instance
   master.start()  # start the master end

   # do something here

   master.shutdown()  # shutdown the master (ATTENTION: master will not be stopped immediately)
   master.join()  # wait until the master completely stopped

   print("master has been stopped")

或通过\ ``with``\ 进行快速调用：

.. code:: python

   with MyMaster('0.0.0.0', 8088, channel=233) as master:  # start the master
       # when this line is reached, all the initialization process has been completed
       # do anything you like here

   print("master has been stopped")  # after quit the master with block, all the resources will be automatically released, and wait until master completely stopped

基于\ ``with``\ 的使用，我们可以通过以下方式进行任务的下达、管理以及结果的获取。结合上文中Slave的例子，举例如下：

.. code:: python

   class MyMaster(Master):
       pass

   if __name__ == '__main__':
   	with MyMaster('0.0.0.0', 8088, channel=233) as master:
           master.ping()  # True if master launch success, otherwise False
           
           with master.new_connection('conn', '127.0.0.1', 8080) as conn:  # establish a connection to slave end
               # when this line is reached, all the initialization process has been completed
               
               assert conn.is_connected  # check if slave connected success
               
               assert 'conn' in master  # check if connection 'conn' still exist and alive in master
               _tmp_conn = master['conn']  # get connection named 'conn' from master
               assert conn == _tmp_conn  # of course, one object actually
               
               task = conn.new_task({'a': 2, 'b': 3})  # create a new task (but has not been sent to slave yet)
               task.start().join()  # start the task and waiting for its completeness            
               assert task.result == {'sum': 5}  # get result of task
               assert task.status == TaskStatus.COMPLETED  # get status of task
               
               task = conn.new_task({'a': 2, 'bb': 3})  # create a new invalid task
               task.start().join()
               assert task.result == {'message': 'ab not found'}  # get result of task failure
               assert task.result == TaskStatus.FAILED  # get status of task
               
               _result_value = None
               def _print_result(result):
                   nonlocal _result_value
                   _result_value = result
               
   			task = conn.new_task({'a': 2, 'b': 3}).on_complete(_print_result)  # create a new task with callback
               task.start().join()
               assert _result_value == {'sum': 5}  # the callback has been triggered
               

此外，还有更多的一些功能和用法，后续会考虑进一步介绍，同时欢迎阅读源代码。

值得注意的有以下几点：

-  为了方便对功能的扩展，\ **请继承Master类，且在需要的时候实现诸如\ ``_before_new_task``\ 等的一系列方法**\ ，不要直接使用Master类创建Master端。

-  **Master和Slave端的channel务必设置为同一个整数**\ ，否则将导致无法正常建立连接。

.. _header-n54:

常见问题
--------

.. _header-n13:

Q：何为非阻塞单线程任务？以及何故作此设计？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A：即为\ **当Master端下达任务时，如果Slave端空闲，则执行任务；若果Slave端已经有一个任务正在运行，则拒绝该任务请求**\ 。

与之类似的也有类似几个任务模式，定义如下：

-  非阻塞多线程任务：Slave端设有最大任务数量，当Master下达任务时，如果正在执行的任务已经达到最大数量，则拒绝新任务请求。

-  阻塞单线程任务：当Master端下达任务时，如果Slave端空闲，则执行任务；若果Slave端已经有一个任务正在运行，则将新任务加入任务队列，等待之前的任务完成后再执行。

-  阻塞多线程任务：Slave端设有最大任务数量，当Master下达任务时，如果正在执行的任务已经达到最大数量，则将新任务加入任务队列，等待之前的任务完成后再执行。

考虑到\ **强化学习训练并发计算量大，不宜在节点上分散算力**\ ，且需要方便业务层调度管理的实际需求，故此处设计为非阻塞单线程任务模式。

.. _header-n122:

Q：Interaction模块适合使用的问题有哪些？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A：实际上根据目前的初步调研，训练任务会分为如下几种情况：

-  **单机轻型**\ 。即在单个具备或不具备GPU的普通计算机上进行训练任务，例如在自己的工作机、笔记本上运行demo。

-  **单机分布式**\ 。即在单个算力较高的计算机或集群上进行训练任务，例如在配备GPU的高配工作站、常见的slurm集群等环境上运行一般的训练任务。

-  **多机规模化分布式**\ 。即在多个计算节点上进行协同训练任务，例如在处于共同内网的100台GPU服务器上运行一个具备规模的训练任务。

实际上对于单机轻型来说，一般的运行即可完成；而\ **对于单机分布式而言，Interaction是完全不必要的**\ ，因为单机分布式环境下完全可以通过fork开启子进程的方式启动各端，并通过进程锁（Lock）和事件（Event）进行阻塞控制，其传输性能和稳定性必然超过基于http服务的Interaction。

因此\ **对于Interaction而言，真正的优势环境为多机规模化分布式环境**\ ，具体来说，因为在多机环境下开启fork或者基于远程启动来启动训练任务是不现实的，因而必须基于Interaction构建服务体系。并且实际上在这样的环境下，\ **最佳实践为预先开启全部的服务节点（即Slave节点），保持长期待机状态，并有专人对这些算力进行维护（类比对slurm集群的维护），当有使用者有训练任务时，会对现有大量节点进行连接，安排并运行训练任务**\ 。

.. _header-n120:

Q：Master和Slave在发送网络请求时出现错误，抛出异常，应该如何处理？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A：抛出的网络请求异常在Interaction框架中，基于错误代码（非HTTP状态码）进行了异常类的归类。在实际使用的时候请\ **注意不要直接使用HTTPError进行异常的捕捉**\ ，该异常只能捕捉非业务异常（例如DNS故障、连接超时等），而对于业务异常，请使用对应的异常类已经捕捉，并且\ **建议根据不同的业务异常类型使用对应的异常类**\ ，以实现精确捕捉和处理问题。

.. _header-n110:

Q：如何正确将Master和Slave整合进现有业务服务中？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A：比较推荐的一种方式——**将Master/Slave作为类的一个私有属性，整合进类内部**\ ，并且\ **建议类本身也对生命周期进行妥善管理**\ （例如设立start、shutdown、join等生命周期管理方法），且建议实现\ ``__enter__``\ 、\ ``__exit__``\ 方法，使得类可以通过\ ``with``\ 进行快速创建和资源回收。

此处\ **强烈不建议直接对Master和Slave类进行二次继承**\ ，因为这会导致Master/Slave本身的结构和生命周期受到干扰，并且影响其内部的逻辑与数据约束，从而造成不可预期的结果。

.. tip::

  这里说的二次继承是指: MyMaster --> Master, Controller --> MyMaster. Controller作为业务逻辑相关的类应和MyMaster是组合关系，切忌滥用继承。如果要为Master做更多功能拓展的话，也可定义相应的功能类，然后MyMaster多重继承Master和新的功能类。
