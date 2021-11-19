How to Use Multi-GPUs to Train Your Model
================================================

DI-engine supports data-parallel training with multi-GPUs.

During data-parallel training, each device handles a portion of total input. 
Large training batch significantly accelerate the training process.

About data-parallel training in DI-engine, we support two types which are DataParallel(DP) and DataDistributedParallel(DDP).

The experimental environment referred to here is a single machine with multi-GPUs.

DataParallel(DP) Mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
DP is mainly used for single-machine multi-GPUs, single-process control multi-GPUs.
There is 1 master node, the default is device[0].
The gradients are aggregated to the master, and the parameters are updated through backpropagation, 
and then the parameters are synchronized with other GPUs.

1. In DI-engine, we define ding.torch_utils.DataParallel, which inherits Torch.nn.DataParallel. And at the same time, we rewrite the parameters() method. please refer to ``ding/torch_utils/dataparallel.py``

.. code-block:: python

    import torch
    import torch.nn as nn

    class DataParallel(nn.DataParallel):
        def __init__(self, module, device_ids=None, output_device=None, dim=0):
            super().__init__(module, device_ids=None, output_device=None, dim=0)
            self.module = module

        def parameters(self, recurse: bool = True):
            return self.module.parameters(recurse = True)

2. Training

.. code-block:: python

    from ding.entry import serial_pipeline
    from ding.model.template.q_learning import DQN
    from ding.torch_utils import DataParallel

    model = DataParallel(DQN(obs_shape=[4, 84, 84],action_shape=6))
    serial_pipeline((main_config, create_config), seed=0, model=model)

We don’t need to change any other code, just simply encapsulate the policy. Please refer to ``dizoo/atari/config/serial/spaceinvaders/spaceinvaders_dqn_config_multi_gpu_dp.py``

For DP, the runnable script demo is demonstrated as follows.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0,1 python -u spaceinvaders_dqn_main_multi_gpu_ddp.py

or (on cluster managed by Slurm)

.. code-block:: bash

    srun -p PARTITION_NAME --mpi=pmi2 --gres=gpu:2 -n1 --ntasks-per-node=1 python -u spaceinvaders_dqn_main_multi_gpu_ddp.py



DataDistributedParallel(DDP) Mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
DataDistributedParallel(DDP) is mainly used for single-machine multi-GPUs and multi-machine multi-GPUs. 
It adopts multi-process to control multi-GPUs and adopts ring allreduce to synchronize gradient.

In DataDistributedParallel(DDP) Mode, we should simply set ``config.policy.learn.multi_gpu`` as `True` in the config file under ``dizoo/atari/config/serial/spaceinvaders/spaceinvaders_dqn_config_multi_gpu_ddp.py``.

We re-implement the data-parallel training module with APIs in ``torch.distributed`` for high scalability.

1. Parameters on Rank-0 GPU are broadcasted to all devices, so that models on different devices share the same initialization.

.. code-block:: python

    def _init_multi_gpu_setting(self, model: torch.nn.Module) -> None:
        for name, param in model.state_dict().items():
            assert isinstance(param.data, torch.Tensor), type(param.data)
            broadcast(param.data, 0)
        for name, param in model.named_parameters():
            setattr(param, 'grad', torch.zeros_like(param))

2. Gradients on different devices should be synchronized after the backward function.

.. code-block:: python

        self._optimizer.zero_grad()
        loss.backward()
        if self._cfg.learn.multi_gpu:
            self.sync_gradients(self._learn_model)
        self._optimizer.step()

.. code-block:: python

    def sync_gradients(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                allreduce(param.grad.data)

Information including loss and reward should be aggregated among devices when applying data-parallel training. DI-engine achieves this with AllReduce operator in a hook, and only saves log files on process with rank 0.
For more related functions, please refer to ``ding/utils/pytorch_ddp_dist_helper.py``

3. Training

When using it, firstly we set ``config.policy.learn.multi_gpu`` as `True` in the config file. Secondly, we need to Initialize the current experimental environment.
Please refer to ``dizoo/atari/entry/spaceinvaders_dqn_main_multi_gpu_ddp.py``

.. code-block:: python

    from ding.utils import DistContext

    with DistContext():
        main(space_invaders_dqn_config,create_config)


For DPP, the runnable script demo is demonstrated as follows.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 spaceinvaders_dqn_main_multi_gpu_ddp.py

or (on cluster managed by Slurm)

.. code-block:: bash

    srun -p PARTITION_NAME --mpi=pmi2 --gres=gpu:2 -n2 --ntasks-per-node=2 python -u spaceinvaders_dqn_main_multi_gpu_ddp.py





