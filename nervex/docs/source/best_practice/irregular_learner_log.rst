How to correctly print irregular log?
================================================


Each policy's ``_forward_learn`` method will return some related dict type info to learner in every iteration, then print the info afterwards.
However, the return dict type info does not always have the same keys.
For example, `Phasic Policy Gradient <https://arxiv.org/pdf/2009.04416.pdf>`_'s auxiliary loss and `TD3 <https://arxiv.org/pdf/1802.09477.pdf>`_'s actor loss will only be calculated at some intervals.

In nerveX, it is permitted to only return some keys in some iterations' info dict. But you must make sure that, in policy's method ``_monitor_vars_learn``, you must list all keys that may appear in info dict. Then learner can utilize sliding window average to print variable's correct value.

Here is an example from TD3.

Every `self._cfg.learn.aux_freq` iterations, will auxiliary losses be calculated once.

.. code:: python

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        if self._train_iteration % self._cfg.learn.aux_freq == 0:
            aux_loss, bc_loss, aux_value_loss = self.learn_aux()
            return {
                # ...
                'aux_value_loss': aux_value_loss,
                'auxiliary_loss': aux_loss,
                'behavioral_cloning_loss': bc_loss,
            }
        else:
            return {
                # ...
            }

But you must list those three losses' names in method ``_monitor_vars_learn``.

.. code:: python

    def _monitor_vars_learn(self) -> List[str]:
        return [
            # ...
            'aux_value_loss',
            'auxiliary_loss',
            'behavioral_cloning_loss',
        ]