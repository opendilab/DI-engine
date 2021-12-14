.. _header-n0:

Interaction Overview
====================

.. _header-n2:

Overview
---------

Interaction module is an interactive service framework independent of any specific business, which used to support simple non-blocking single-thread tasks. It provides support for training(multi-machine multi-card) in a distributed environment based on ding.

Interaction module is divided into two ends, the ``Master`` and the ``Slave``, which are the task release end and the task execution end respectively.


.. _header-n9:

Slave
------

``Slave`` receives task information from ``Master``, executes the task by rewriting the ``_process_task`` method, and returns the message through the return value and exception raising.

Here is a simple example for building your own ``Slave`` class by following way:

.. code:: python

   class MySlave(Slave):  # build a slave class, inheriting from Slave base class
       def _process_task(self, task: Mapping[str, Any]):
           if 'a' in task.keys() and 'b' in task.keys():
               return {'sum': task['a'] + task['b']}
           else:
               raise TaskFail(result={'message': 'ab not found'}, message='A or B not found in task data.')

The logical of this class is that if all parameters ``a`` and ``b`` exist, the sum of the two is returned in json format, otherwise the ``TaskFail`` exception will be  raised to indicates that the task execution failed.

There are two ways to call the ``Slave`` class. The first is to start and close the ``Slave`` in a normal way:

.. code:: python

   slave = MySlave('0.0.0.0', 8080, channel=233)  # instantiate new slave instance
   slave.start()  # start the slave end

   # do something here

   slave.shutdown()  # shutdown the slave (ATTENTION: slave will not be stopped immediately after this)
   slave.join()  # wait until the slave completely stopped

   print("slave has been stopped")

Or using ``with`` for quick call:

.. code:: python

   with MySlave('0.0.0.0', 8080, channel=233):  # start the slave
       # when this line is reached, all the initialization process has been completed
       run_until_ctrl_c()  # just block here

   print("slave has been stopped")  # after quit the slave with block, all the resources will be automatically released, and wait until slave completely stopped

After the calling above, connecting port(``8080``) of the slave on the master and select channel ``233`` to complete the connection.

Attention:

- In order to facilitate the realization of the task logic clearly, **please inherit the Slave class and implement the _task_process method**. Do not directly use the Slave class to create the Slave end.

- Channel can be regarded as a kind of identification for different specific services, and its concept is similar to radio channel, network port, etc. **It is recommended to set a specific channel value for a specific task**. Do not directly use a value that is easy to collide, such as 0 or the default, to ensure that you can get feedback immediately when a misconnection occurs.

.. _header-n57:

Master
--------

The main job of the ``Master`` is to connect to the ``Slave``, and to issue tasks to the ``Slave`` through the established connection.

Here is a simple example to build your own ``Master`` class in the following way:

.. code:: python

   class MyMaster(Master):  # build a master class, inherit from master base class
       pass

Similar to ``Slave``, there are two ways to start the ``Master``. The first is to start and stop the ``Master`` in the common way:

.. code:: python

   master = MyMaster('0.0.0.0', 8088, channel=233)  # instantiate new master instance
   master.start()  # start the master end

   # do something here

   master.shutdown()  # shutdown the master (ATTENTION: master will not be stopped immediately)
   master.join()  # wait until the master completely stopped

   print("master has been stopped")

Or make a quick call using ``with``:

.. code:: python

   with MyMaster('0.0.0.0', 8088, channel=233) as master:  # start the master
       # when this line is reached, all the initialization process has been completed
       # do anything you like here

   print("master has been stopped")  # after quit the master with block, all the resources will be automatically released, and wait until master completely stopped

By using ``with``, we can issue tasks, manage tasks, and obtain results in the following ways in ``Master``. Combining the ``Slave`` example above, an example is as follows:

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


In addition, there are more functions and usages, which will be further introduced in the follow-up, and welcome to read the source code.

Attentions:

- In order to make it convient for the extension of functions, please inherit the Master class and implement a series of methods such as ``_before_new_task`` when needed. Do not directly use the ``Master`` class to create the ``Master``.

- **The channel of the Master and Slave must be set to the same**, otherwise the connection cannot be established normally.


.. _header-n54:

Q & A
--------

.. _header-n13:

Q: What is a non-blocking single-threaded task? Why make this design?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A: That is, **when Master issues a task, if the Slave is idle, the task will be executed; if there is already a task running on the Slave, the task request will be rejected**.

Similarly, there are several task modes, which are defined as follows:

- Non-blocking multi-thread task: The ``Slave`` has a maximum number of tasks to excute. When the ``Master`` issues a task, if the number of tasks being executed has reached the maximum number, the new task request will be rejected.

- Blocking single-thread task: When the ``Master`` sends a task, if the ``Slave`` is idle, the task will be executed; if there is already a task running on the Slave side, the new task will be added to the task queue, and the task will be executed after the previous task is completed.

- Blocking multi-thread task: The ``Slave`` has a maximum number of tasks. When the Master issues a task, if the number of tasks being executed has reached the maximum number, the new task will be added to the task queue and wait for the completion of the previous task before executing.

Considering **the large amount of concurrent computing in reinforcement learning training, it is not appropriate to disperse computing power on nodes**, and the actual needs of facilitating business-level scheduling management, so the design here is switched to non-blocking single-threaded task mode.

.. _header-n122:

Q: What are the issues that the Interaction module is suitable for use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A: Actually, according to the current preliminary investigation, the training tasks will be divided into the following situations:

- **Stand-alone light**. That is, training tasks are performed on a single ordinary computer with or without GPU, such as running demos on their own work machines and laptops.

- **Stand-alone distributed**. That is, training tasks are performed on a single computer or cluster with higher computing power. For example, general training tasks are run on environments such as workstations equipped with GPUs and common slurm clusters.

- **Multi-machine large-scale distributed**. That is, a collaborative training task is performed on multiple computing nodes. For example, a large-scale training task is run on 100 GPU servers in a common intranet.

In fact, for a single-machine light-duty, the general operation can be completed. For a single-machine distributed, Interaction is completely unnecessary, because in a single-machine distributed environment, it is possible to start each end by forking the child processes, and do blocking control through ``Lock`` and ``Event``. Its transmission performance and stability will inevitably exceed the HTTP service-based Interaction.

Therefore, **for Interaction, the best environment to show its advantage is a multi-machine large-scale distributed environment**. Specifically, because it is unrealistic to start a fork in a multi-machine environment or start training tasks based on remote startup, it is necessary to build services based on Interaction. In fact, in such an environment, **the best practice is to turn on all service nodes (Slave nodes) in advance, keep them in a long-term standby state, and have dedicated personnel to maintain these computing nodes (analogous to the maintenance of the slurm cluster). When users have training tasks, they will connect a large number of existing nodes, arrange and run training tasks**.

.. _header-n120:

Q: What should I do if an error occurs when the Master and Slave are sending network requests?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A: The network exception thrown is classified in the Interaction framework based on the error code (non-HTTP status code). In actual use, **please be careful not to directly use HTTPError to capture exceptions**. This exception can only capture non-business exceptions (such as DNS failures, connection timeouts, etc.). For business exceptions, please use the corresponding exception class that has been captured, and **it is recommended to use corresponding exception classes according to different business exception types** to accurately capture and handle problems.

.. _header-n110:

Q: How to correctly integrate Master and Slave into existing business services?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A: A more recommended way is **to treat Master/Slave as a private attribute of the class and integrated into the class**, and **it is also recommended to properly manage the life cycle for the class itself** (for example, set up start, shutdown, join and other life cycle management methods ), and it is recommended to implement the ``__enter__`` and ``__exit__`` methods, so that the class can be quickly created and recycled through ``with``.

**It is strongly not recommended to directly carry out secondary inheritance of the Master and Slave classes**, because this will cause the structure and life cycle of the Master/Slave itself to be disturbed, and affect its internal logic and data constraints, resulting in unpredictable results.

.. tip::

    The secondary inheritance mentioned here refers to: ``MyMaster`` --> ``Master``, ``Controller`` --> ``MyMaster``. ``Controller`` as a class related to business logic should have a composite relationship with ``MyMaster``, and avoid abuse of inheritance. If you want to expand more functions for ``Master``, you can also define corresponding function classes, and then let ``MyMaster`` multiple inherits ``Master`` and new function classes.
