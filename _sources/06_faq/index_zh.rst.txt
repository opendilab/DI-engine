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


Q6: 如何理解打印出的 [EVALUATOR] 信息
**************************************************

:A6:

我们在 `interaction_serial_evaluator.py <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/interaction_serial_evaluator.py#L253>`_ ，打印出 ``evaluator`` 的评估信息，
包括 ``env``, ``final reward``, ``current episode`` 分别代表当前已完成局 (``timestep.done=True``) 对应的环境索引 (``env_id``), 当前已完成局游戏的奖励， 以及它是 ``evaluator`` 评估的第几局游戏。
一个典型的 evaluator log 信息如下图所示：

.. image:: images/Q6_evaluator_info.png
   :align: center
   :height: 250

在某些情况下，``evaluator`` 中的不同环境可能会收集不同长度的游戏局。 例如，假设我们通过 ``evaluator`` 收集 16 局游戏，但只有 5 个评估环境 (``eval_env``)，即在 config 中设置 ``n_evaluator_episode=16, evaluator_env_num=5``，
我们如果不对每个评估环境的评估总局数进行限制，很可能会得到许多步长较短的游戏局，这样一来，这次评估阶段得到的平均奖励就会有偏差，不能完全反映当前 policy 的性能 (只反映了在步数较短的游戏局上的性能)。

我们通过使用 `VectorEvalMonitor <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/base_serial_evaluator.py#L78>`_ 类来缓解这个问题。
在这个类中，我们在 `这里 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/base_serial_evaluator.py#L103>`_ 平均指定每个 ``eval_env`` 需要评估的局数，
例如，如果设置 ``n_evaluator_episode=16`` 和 ``evaluator_env_num=8``，那么每个 ``eval_env`` 只有 2 局将被添加到统计量中。
关于 ``VectorEvalMonitor`` 每个方法的具体含义，请参考类 `VectorEvalMonitor <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/base_serial_evaluator.py#L78>`_ 的注释。

..
    通过 `dict <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/base_serial_evaluator.py#L110>`_ 来存储在各个 ``eval_env`` 上运行的游戏局的 reward, 注意这里对于每个 ``eval_env`` 是用一个
    ``deque`` 来存储reward的 (指定 ``max_length`` 等于 ``每个eval_env需要评估的局数`` (在代码中为 ``each_env_episode[i]`` )）。
    我们通过 `update_reward <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/base_serial_evaluator.py#L133>`_ 方法根据 ``env_id`` 来更新每个环境已评估局的reward。

值得注意的是，当 evaluator 的某一个 ``eval_env`` 完成评估数量为 ``each_env_episode[i]`` 的游戏局后，由于环境的 reset 是由
`env_manager <https://github.com/opendilab/DI-engine/blob/main/ding/envs/env_manager/subprocess_env_manager.py>`_  自动控制的，它还会继续运行下去, 除非整个评估阶段终止。
我们是用 ``VectorEvalMonitor`` 控制评估阶段的终止，只有当
`eval_monitor.is_finished() <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/interaction_serial_evaluator.py#L224>`_ 为True时，
即 evaluator 完成了所有的评估任务后 (在所有 ``eval_env`` 上一共评估了 ``n_evaluator_episode`` 局)，才会退出本次评估, 所以可能会出现某个 ``eval_env`` 在完成它自己的评估数量为 ``each_env_episode[i]`` 的游戏局后，其对应的log信息仍然重复出现的情况,
用户不必担心这些重复的 logs，它们不会对评估结果产生不好的影响。