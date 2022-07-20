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


Q5: 如何调整学习率
**************************************************

:A5:

在相应算法的入口添加 ``lr_scheduler`` 代码，可以通过调用 ``torch.optim.lr_scheduler`` （参考： `<https://pytorch.org/docs/stable/optim.html>`_）模块来调整学习率，并且在模型优化更新之后使用 ``scheduler.step()`` 对学习率进行更新。
下面代码提供一个简单的案例，具体可参考demo: `<https://github.com/opendilab/DI-engine/commit/9cad6575e5c00036aba6419f95cdce0e7342630f>`_。

.. code::

    from torch.optim.lr_scheduler import LambdaLR

    ...

    # Set up RL Policy
    policy = DDPGPolicy(cfg.policy, model=model)
    # Set up lr_scheduler, the optimizer attribute will be different in different policy.
    # For example, in DDPGPolicy the attribute is 'optimizer_actor', but in DQNPolicy the attribute is 'optimizer'.
    lr_scheduler = LambdaLR(
        policy.learn_mode.get_attribute('optimizer_actor'), lr_lambda=lambda iters: min(1.0, 0.5 + 0.5 * iters / 1000)
    )

    ...

    # Train
        for i in range(cfg.policy.learn.update_per_collect):
            ...
            learner.train(train_data, collector.envstep)
            lr_scheduler.step()

学习率变化曲线如图所示

.. image:: images/Q5_lr_scheduler.png
   :align: center
   :height: 250
