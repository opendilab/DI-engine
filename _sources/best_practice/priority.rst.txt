How to Use PER(Prioritized Experience Replay)
================================================

Guideline
^^^^^^^^^^
Prioritized Sample is an important mechanism in some algorithms, for example, Rainbow.
It inclues:

   1. Sample according to priority, instead of traditional uniform sample.
   2. Calculate Importance Sampling Weight as each data's loss weight.
   3. After a train iteration, update priority in buffer

Procedures
^^^^^^^^^^^^^

1. Open priority mechanism and set corresponding hyper-parameters in config

   ``priority`` must be set to ``True``. ``priority_IS_weight`` means whether to use IS weight to correct the bias. It is recommended to set to ``True``, but you can feel free not to use it.

   ``alpha``, ``beta``, ``anneal_step`` are hyper-parameters in priority mechanism.

   .. code:: python

      policy=dict(
          ...,
          priority=True,
          priority_IS_weight=True,
          ...,
          other=dict(
            replay_buffer=dict(
               ...,
               # How much priority is used.
               alpha=0.6,
               # How much correction is used.
               beta=0.4,
               # Beta annealing. Sample step count.
               anneal_step=0,
            )
          ),
      )

2. Use Importance Sampling as loss weight

   ``PrioritizedBufferr`` would sample data with probabilities proportional to their priorities. And it would also add a key-value pair ``IS`` into data dict. ``IS`` is "Importance Sampling Weight", which is used to correct the biased optimization process caused by prioritized sampling. Each sampled data's loss will multiply corresponding weight respctively if ``priority_IS_weight`` is ``True``.

      .. code:: python

         import torch.nn.functional as F

         # tensor shape: output (B, ), target (B, )
         # not use IS
         loss = F.mse_loss(output, target)
         # use IS (recommended)
         loss = (F.mse_loss(output, target, reduction='none') * data['IS']).mean()
         # DI-engine td error(data['weight'] = data['IS'], assigned in policy._forward_learn method)

3. Update priority in buffer

   Since priority is a by-product of error calculation, you can directly get new priority in method ``policy._forward_learn``. Then you can add the key-value pair to the return dict. Make sure that its key is ``"priority"``, its value is a ``list`` with length "batch_size".

   .. code:: python

      data_n = q_nstep_td_data(
          q_value, target_q_value, data['action'], target_q_action, reward, data['done'], data['weight']
      )
      loss, td_error_per_sample = q_nstep_td_error(data_n, self._gamma, nstep=self._nstep)
      return {
          'total_loss': loss.item(),
          'priority': td_error_per_sample.abs().tolist(),
      }

Others
^^^^^^^

1. Calculate initial priority in collectors

    Usually, priority is initialized when this data is inserted into replay buffer with default value or the maximum history priority value, DI-engine also supports priority calculation and initialization in collector:

      - Method ``policy._forward_collect`` will calculate priority as well，and return the key-value pair.
      - Method ``policy._process_transition`` will put ``model_output['priority']`` into returned data, as its initial priority.

    .. code:: python

        def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
            transition = {
                'obs': obs,
                'next_obs': timestep.obs,
                'action': model_output['action'],
                'priority': model_output['priority'],  # add this one
                'reward': timestep.reward,
                'done': timestep.done,
            }
            return transition

2. Different exploration strategies
   
   In Ape-X, different collectors can use different exploration strategies(e.g.: different epsilon values for different collectors). Now DI-engine also supports this mechanism. In serial pipeline, you need to implement your own main entry function to control when to change exploration strategies,
   and override ``policy._forward_colleect`` method to receive control arguments and execute the corresponding strategy. In parallel entry, you should set different parameters in commander for different collectors.
