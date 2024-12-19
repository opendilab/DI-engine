Using DDP Distributed Training in DI-engine
===========================================

Distributed Data Parallel (DDP) is an effective approach to accelerate reinforcement learning training. This document provides detailed instructions on configuring and using DDP training in DI-engine, using the example ``pong_dqn_ddp_config.py``.

Launching DDP Training
----------------------

To launch DDP training, use PyTorch's ``distributed.launch`` module. Run the following command:

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 ./dizoo/atari/config/serial/pong/pong_dqn_ddp_config.py

Where:

- ``--nproc_per_node=2``: Specifies the use of 2 GPUs for training.
- ``--master_port=29501``: Specifies the port number for the master process.
- The final argument is the path to the configuration file.

Configuration for DDP Training
------------------------------

Compared to single-GPU training (e.g., ``pong_dqn_config.py``), the configuration file ``pong_dqn_ddp_config.py`` for enabling multi-GPU DDP training introduces two key changes:

1. **Enable multi-GPU support in the policy configuration**:

   .. code-block:: python

       policy = dict(
           multi_gpu=True,  # Enable multi-GPU training mode
           cuda=True,       # Use CUDA acceleration
           ...
       )

   - The core code for initializing multi-GPU training is located in ``base_policy.py``:
     `base_policy.py#L167 <https://github.com/opendilab/DI-engine/blob/main/ding/policy/base_policy.py#L167>`_
   - Gradient synchronization occurs in ``policy._forward_learn()``:
     `dqn.py#L281 <https://github.com/opendilab/DI-engine/blob/main/ding/policy/dqn.py#L281>`_

2. **Use ``DDPContext`` to manage the distributed training process**:

   .. code-block:: python

       if __name__ == '__main__':
           from ding.utils import DDPContext
           from ding.entry import serial_pipeline
           with DDPContext():
               serial_pipeline((main_config, create_config), seed=0, max_env_step=int(3e6))

   - ``DDPContext`` initializes the distributed training environment and releases distributed resources after training completes.

DDP Implementation in DI-engine
--------------------------------

The DDP implementation in DI-engine includes the following key components:

1. **Distributed Processing of Data Collection in the Collector**:

   - In ``SampleSerialCollector``, each process independently collects data samples.
   - After collection, statistical data is synchronized across processes using ``allreduce``:

     .. code-block:: python

         if self._world_size > 1:
             collected_sample = allreduce_data(collected_sample, 'sum')
             collected_step = allreduce_data(collected_step, 'sum')
             collected_episode = allreduce_data(collected_episode, 'sum')
             collected_duration = allreduce_data(collected_duration, 'sum')

     - See `sample_serial_collector.py#L355 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/sample_serial_collector.py#L355>`_

2. **Distributed Processing of the Evaluator**:

   - The evaluation logic runs only on the rank 0 process:

     .. code-block:: python

         if get_rank() == 0:
             # Perform evaluation logic
             ...

     - See `interaction_serial_evaluator.py#L207 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/interaction_serial_evaluator.py#L207>`_

   - After evaluation, the results are broadcast to other processes:

     .. code-block:: python

         if get_world_size() > 1:
             objects = [stop_flag, episode_info]
             broadcast_object_list(objects, src=0)
             stop_flag, episode_info = objects

     - See `interaction_serial_evaluator.py#L315 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/interaction_serial_evaluator.py#L315>`_

3. **Distributed Logging**:

   - The logger is initialized only on the rank 0 process:

     .. code-block:: python

         tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None

     - See `serial_entry.py#L72 <https://github.com/opendilab/DI-engine/blob/main/ding/entry/serial_entry.py#L72>`_

   - Logging is restricted to the rank 0 process:

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

     - See `sample_serial_collector.py#L59 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/sample_serial_collector.py#L59>`_

   - Printing logs is also restricted to the rank 0 process:

     .. code-block:: python

         if self._rank != 0:
             return

     - See `sample_serial_collector.py#L388 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/sample_serial_collector.py#L388>`_

Summary
-------

In DI-engine, DDP distributed training fully utilizes the computational power of multiple GPUs to accelerate the training process through distributed data collection, evaluation, and logging. The core logic of DDP relies on PyTorch's distributed framework, while ``DDPContext`` provides unified management of the distributed environment, simplifying the configuration and usage process for developers.

For more details on the implementation, refer to the following code references:

- `base_policy.py#L167 <https://github.com/opendilab/DI-engine/blob/main/ding/policy/base_policy.py#L167>`_
- `dqn.py#L281 <https://github.com/opendilab/DI-engine/blob/main/ding/policy/dqn.py#L281>`_
- `serial_entry.py#L72 <https://github.com/opendilab/DI-engine/blob/main/ding/entry/serial_entry.py#L72>`_
- `sample_serial_collector.py#L355 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/sample_serial_collector.py#L355>`_
- `sample_serial_collector.py#L59 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/sample_serial_collector.py#L59>`_
- `sample_serial_collector.py#L388 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/sample_serial_collector.py#L388>`_
- `interaction_serial_evaluator.py#L207 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/interaction_serial_evaluator.py#L207>`_
- `interaction_serial_evaluator.py#L315 <https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/interaction_serial_evaluator.py#L315>`_