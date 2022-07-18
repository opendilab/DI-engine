FAQ
=====================

.. toctree::
   :maxdepth: 1

Q1: 关于使用时出现的warning
****************************

:A1:

对于运行DI-engine时命令行中显示的import linlink, ceph, memcache, redis的相关warning，一般使用者忽略即可，DI-engine会在import时自动进行寻找相应的替代库或代码实现。


Q2: 安装之后无法使用DI-engine命令行工具(CLI)
********************************************

:A2:

- 部分运行环境使用pip安装时指定 ``-e`` 选项会导致无法使用CLI，一般非开发者无需指定该选项，去掉该选项重新安装即可
- 部分运行环境会将CLI安装在用户目录下，需要验证CLI的安装目录是否在使用者的环境变量中 （如 linux 的 ``$PATH`` 中）


Q3: 安装时出现"没有权限"相关错误
***********************************

:A3:

由于某些运行环境中缺少相应权限，pip安装时可能出现"没有权限"(Permission denied)，具体原因及解决方法如下：
 - pip添加 ``--user`` 选项，安装在用户目录下
 - 将仓库根目录下的 ``.git`` 文件夹移动出去，执行pip安装命令，再将其移动回来，具体原因可参见  `<https://github.com/pypa/pip/issues/4525>`_


Q4: 如何设置SyncSubprocessEnvManager的相关运行参数
**************************************************

:A4:

在配置文件的env字段添加manager字段，可以指定是否使用shared_memory，多进程multiprocessing启动的上下文，下面的代码提供了一个简单样例，详细的参数信息可参考 ``SyncSubprocessEnvManager``

.. code::

    config = dict(
        env=dict(
            manager=dict(shared_memory=False)
        )
    )
