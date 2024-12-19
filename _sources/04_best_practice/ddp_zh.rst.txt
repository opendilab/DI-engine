使用 DI-engine 中的 DDP 分布式训练
==================================

分布式数据并行（Distributed Data Parallel, DDP）是一种有效提升强化学习训练效率的技术。本指南以示例配置文件 ``pong_dqn_ddp_config.py`` 为例，详细说明如何在 DI-engine 框架中配置和使用 DDP 进行分布式训练。

启动 DDP 训练
--------------

要启动 DDP 训练，可以使用 PyTorch 的 ``torch.distributed.launch`` 模块，运行以下命令：

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 ./dizoo/atari/config/serial/pong/pong_dqn_ddp_config.py

其中：

- ``--nproc_per_node=2``：指定使用 2 个 GPU 进行训练。
- ``--master_port=29501``：指定主进程的通信端口号。
- 最后的参数是训练配置文件的路径。

DDP 训练配置
-------------

与单 GPU 训练配置文件（如 ``pong_dqn_config.py``）相比，基于 DDP 的多 GPU 训练配置文件 ``pong_dqn_ddp_config.py`` 主要新增了以下两处改动：

1. **在策略配置中启用多 GPU 支持：**

   .. code-block:: python

       policy = dict(
           multi_gpu=True,  # 启用多 GPU 训练模式
           cuda=True,       # 使用 CUDA 加速
           ...
       )

   - 多 GPU 训练的核心逻辑实现于 ``base_policy.py``：
     `base_policy.py#L167 <https://github.com/opendilab/DI-engine/blob/main/ding/policy/base_policy.py#L167>`_
   - 梯度同步的实现位于 ``policy._forward_learn()``：
     `dqn.py#L281 <https://github.com/opendilab/DI-engine/blob/main/ding/policy/dqn.py#L281>`_

2. **使用 ``DDPContext`` 管理分布式训练环境：**

   .. code-block:: python

       if __name__ == '__main__':
           from ding.utils import DDPContext
           from ding.entry import serial_pipeline
           with DDPContext():
               serial_pipeline((main_config, create_config), seed=0, max_env_step=int(3e6))

   - ``DDPContext`` 负责初始化分布式环境，并在训练结束后释放相关资源。

DI-engine 中的 DDP 实现
------------------------

DI-engine 的 DDP 训练框架由以下核心部分组成：

1. **分布式数据收集：**

   - 在 ``SampleSerialCollector`` 中，每个进程独立收集数据样本。
   - 数据收集完成后，通过 ``allreduce`` 同步统计结果：

     .. code-block:: python

         if self._world_size > 1:
             collected_sample = allreduce_data(collected_sample, 'sum')
             collected_step = allreduce_data(collected_step, 'sum')
             collected_episode = allreduce_data(collected_episode, 'sum')
             collected_duration = allreduce_data(collected_duration, 'sum')

     - 代码位置：
       `sample_serial_collector.py#L355 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/sample_serial_collector.py#L355>`_

2. **分布式评估：**

   - 评估逻辑仅在 ``rank 0`` 进程上运行：

     .. code-block:: python

         if get_rank() == 0:
             # 执行评估逻辑
             ...

     - 代码位置：
       `interaction_serial_evaluator.py#L207 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/interaction_serial_evaluator.py#L207>`_

   - 评估完成后，结果通过广播同步至其他进程：

     .. code-block:: python

         if get_world_size() > 1:
             objects = [stop_flag, episode_info]
             broadcast_object_list(objects, src=0)
             stop_flag, episode_info = objects

     - 代码位置：
       `interaction_serial_evaluator.py#L315 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/interaction_serial_evaluator.py#L315>`_

3. **分布式日志记录：**

   - 日志记录器仅在 ``rank 0`` 进程上初始化：

     .. code-block:: python

         tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None

     - 代码位置：
       `serial_entry.py#L72 <https://github.com/opendilab/DI-engine/blob/main/ding/entry/serial_entry.py#L72>`_

   - 日志记录仅限 ``rank 0`` 进程：

     .. code-block:: python

         if self._rank == 0:
             if tb_logger is not None:
                 self._logger, _ = build_logger(
                     path='./{}/log/{}'.format(self._exp_name, self._instance_name),
                     name=self._instance_name,
                     need_tb=False
                 )
                 self._tb_logger = tb_logger
             else:
                 self._logger, self._tb_logger = build_logger(
                     path='./{}/log/{}'.format(self._exp_name, self._instance_name),
                     name=self._instance_name
                 )
         else:
             self._logger, _ = build_logger(
                 path='./{}/log/{}'.format(self._exp_name, self._instance_name),
                 name=self._instance_name,
                 need_tb=False
             )
             self._tb_logger = None

     - 代码位置：
       `sample_serial_collector.py#L59 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/sample_serial_collector.py#L59>`_

   - 打印日志也仅限于 ``rank 0`` 进程：

     .. code-block:: python

         if self._rank != 0:
             return

     - 代码位置：
       `sample_serial_collector.py#L388 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/sample_serial_collector.py#L388>`_

总结
----

DI-engine 的 DDP 分布式训练能够充分利用多 GPU 的计算能力，通过分布式的数据收集、评估和日志记录有效加速训练过程。其核心逻辑基于 PyTorch 的分布式框架，而 ``DDPContext`` 提供了简洁易用的分布式环境管理接口，显著降低了开发者的配置难度。

有关实现的更多细节，请参考以下代码位置：

- `base_policy.py#L167 <https://github.com/opendilab/DI-engine/blob/main/ding/policy/base_policy.py#L167>`_
- `dqn.py#L281 <https://github.com/opendilab/DI-engine/blob/main/ding/policy/dqn.py#L281>`_
- `serial_entry.py#L72 <https://github.com/opendilab/DI-engine/blob/main/ding/entry/serial_entry.py#L72>`_
- `sample_serial_collector.py#L355 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/sample_serial_collector.py#L355>`_
- `sample_serial_collector.py#L59 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/sample_serial_collector.py#L59>`_
- `sample_serial_collector.py#L388 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/sample_serial_collector.py#L388>`_
- `interaction_serial_evaluator.py#L207 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/interaction_serial_evaluator.py#L207>`_
- `interaction_serial_evaluator.py#L315 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/interaction_serial_evaluator.py#L315>`_
