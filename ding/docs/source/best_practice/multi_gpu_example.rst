How to Use Multi-GPU to Train Your Model
================================================

DI-engine supports data-parallel training with multiple GPUs.

During data-parallel training, each device handles a portion of total input. Large training batch significantly accelerate the training process.

To enable multi-gpu training, you can simply set ``config.learn.multi_gpu`` as `True` in the policy file under ``ding/policy/``.

Going deeper with multi-gpu training in DI-engine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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