How to use RNN
==============

Introduction to RNN
-------------------

Recurrent neural network (RNN) is a class of neural network where
connections between nodes form a directed graph along a temporal
sequence. This allows it to exhibit temporal dynamic behavior. Derived
from feedforward neural networks, RNNs can use their internal state
(memory) to process variable length sequences of inputs. This makes them
applicable to tasks such as unsegmented, connected handwriting
recognition or speech recognition.

In deep reinforcement learning, RNN is first used in DRQN(Deep Recurrent
Q-Learning Network), which aims to solve the problem of paritial
observation in atari games. After that, RNN has become an important
method to solve the environments of complex temporal dependence.

After many years of research, RNN has many variants like LSTM, GRU, etc.
The core update process still remains similar. In every timestep
:math:`t` in MDP, agent needs observation :math:`s_t` and historical
observations :math:`s_{t-1}, s_{t-2}, ...` to infer :math:`a_t`. This
requires RNN agent to hold previous observations and maintain RNN hidden
states.

DI-engine supports for RNN , and provides easy to use API to allow users to
implement variants of RNN.

Related Components in DI-engine
--------------------------------

1. ``ding/model/wrapper/model_wrappers.py: HiddenStateWrapper`` :
   Used to maintain hidden states

2. ``ding/torch_utils/network/rnn.py``: Used to build RNN model

3. ``ding/rl_utils/adder.py: Adder:``: Used to arrange origin data into
   time sequence data(by calling ``ding/utils/default_helper.py: list_split()`` function)

RNN example in DI-engine
--------------------------

======= ===========
policy  RNN-support
======= ===========
a2c     ×
atoc    ×
c51     ×
collaq  √
coma    √
ddpg    ×
dqn     ×
il      ×
impala  ×
iqn     ×
ppg     ×
ppo     ×
qmix    √
qrdqn   ×
r2d2    √
rainbow ×
sac     ×
sqn     × 
======= ===========

Use RNN in DI-engine can be described as the following precedures.

-  Build your RNN model

-  Wrap you model in policy

-  Arrange original data to time sequence

-  Initialize hidden state

-  Burn-in(Optional)

Build a Model with RNN
~~~~~~~~~~~~~~~~~~~~~~

You can use either DI-engine's built-in recurrent model or your own RNN
model.

1. Use DI-engine's built-in model. DI-engine's DRQN provide RNN
   support(default to LSTM) for discrete action space environments. You
   can easily specify model type in config or set model in policy to use
   it.

.. code:: python

   # in config file
   policy=dict(
       ...
       model=dict(
         type='drqn',
         import_names=['ding.model.template.q_learning']
       ),
       ...
   ),
   ...

   # or set policy default model
     def default_model(self) -> Tuple[str, List[str]]:
         return 'drqn', ['ding.model.template.q_learning']

2. Use customized model. To use customized model, you can refer to `Set
   up Policy and NN
   model <..//quick_start/index.html#set-up-policy-and-nn-model>`_.
   To adapt your model into DI-engine's pipline with minimal code changes,
   the output dict of model should contain ``'next_state'`` key.

.. code:: python

   class your_rnn_model(nn.Module):

     def forward(x):
         # the input data `x` must be a dict, contains the key 'prev_state', the hidden state of last timestep
         ...
         return {
             'logit': logit,
             'next_state': hidden_state,
             ...
         }

.. note::
   DI-engine also provide RNN module. You can use ``get_lstm()`` function by ``from ding.torch_utils import get_lstm``. This function allows users to build LSTM implemented by ding/pytorch/HPC.


.. _use-model-wrapper-to-wrap-your-rnn-model-in--policy:

Use model wrapper to wrap your RNN model in policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As RNN model need to maintain hidden state of data, DI-engine provide
``HiddenStateWrapper`` for it. Users only need to add a wrapper in
policy's learn/collect/eval initialization to wrap model. The wrapper
will help agent to keep hidden states after model forward and send
hidden states to model in next time forward.

.. code:: python

   # In policy
   class your_policy(Policy):

       def _init_learn(self) -> None:
           ...
         	self._learn_model = model_wrap(self._model, wrapper_name='hidden_state', state_num=self._cfg.learn.batch_size)

   	def _init_collect(self) -> None:
           ...
           self._collect_model = model_wrap(
           self._model, wrapper_name='hidden_state', state_num=self._cfg.collect.env_num, save_prev_state=True
       )

   	def _init_eval(self) -> None:
       	...
           self._eval_model = model_wrap(self._model, wrapper_name='hidden_state', state_num=self._cfg.eval.env_num)

.. note::
   Set ``save_prev_state=True`` in collect model's wrapper to make sure there is previous hidden state for learner to initialize RNN.

More details of ``HiddenStateWrapper`` can be found in `model
wrapper <./model_wrapper.rst>`__, the work flow of it can be shown as
the following figure:

        .. image:: images/model_hiddenwrapper_img.png
            :align: center
            :scale: 60%

Data Arrangement
~~~~~~~~~~~~~~~~

The mini-batch data used for RNN is different from usual RL data, it
should be arranged in time series. For DI-engine, this process happens in
``collector``. Users need to specify ``unroll_len`` in config to make
sure the length of sequence data matches your algorithm. For most cases,
``unroll_len`` should be equal to RNN's historical length. For example,
the original sampled data is :math:`[x_1,x_2,x_3,x_4,x_5,x_6]`, each
:math:`x` represents :math:`[s_t,a_t,r_t,d_t,s_{t+1}]` (maybe
:math:`log_\pi(a_t|s_t)`, hidden state, etc in it), and we need RNN's
historical length to be 3. By specify ``unroll_len=3``, the data will be
arranged as :math:`[[x_1,x_2,x_3],[x_4,x_5,x_6]]`.

If the ``unroll_len`` is not divided by ``n_sample`` of collector, the
residual data will be filled by last sample, i.e. if ``n_sample=6`` and
``unroll_len=4``, the data will be arranged as
:math:`[[x_1,x_2,x_3,x_4],[x_5,x_6,x_6,x_6]]` by default. DI-engine's
``get_train_sample`` have ``drop`` and ``null_padding`` method for this case, to
use it, you need to specify the arguments of ``get_train_sample`` method in policy's collect related method.

For ``drop``, it means data'll be arranged as :math:`[[x_1,x_2,x_3,x_4]]`,
For ``null_padding``, it means data'll be arranged as :math:`[[x_1,x_2,x_3,x_4],[x_5,x_6,x_{null},x_{null}]]`,
:math:`x_{null}` is similar to :math:`x_6` but its ``done=True`` and ``reward=0``. More details can be found in `Adder <../api_doc/rl_utils/adder.html?highlight=adder#ding.rl_utils.adder.Adder>`_.

Initialize Hidden State
~~~~~~~~~~~~~~~~~~~~~~~

The `_learn_model` of policy needs to initialize RNN. These hidden states comes from `prev_state` saved by `_collect_model`.
Users need to add these states to `_learn_model` input data dict by `_process_transition` function.

.. code:: python

   def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:

        transition = {
            'obs': obs,
            'action': model_output['action'],
            'prev_state': model_output['prev_state'], # add `prev_state` key here
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

Then in `_learn_model` forward function, call its reset function(overwritten by HiddenStateWrapper) to initialize RNN with data's
`prev_state`.

.. code:: python

   def _forward_learn(self, data: dict) -> Dict[str, Any]:
        # forward
        data = self._data_preprocess_learn(data)
        self._learn_model.train()
        self._learn_model.reset(data_id=None, state=data['prev_state'][0])


Burn-in(Optional)
~~~~~~~~~~~~~~~~~

This concept comes from R2D2(Recurrent Experience Replay in Distributed
Reinforcement Learning). When using LSTM, we either use a zero start
state to initialize the network at the beginning of sampled sequences,
or replay whole episode trajectories. The former brings bias and the
latter is hard to implement. 

Burn-in allow the network a
``burn-in period`` by using a portion of the replay sequenceonly for
unrolling the network and producing a start state, and update the
network only onthe remaining part of the sequence. In DI-engine, to
implement ``burn-in``, ``unroll_len`` should be set to
``burnin_step+1``\ (if use n-step return, it should be
``burnin_step+2*n_steps``). In this setting, the unrolled data is split
into ``burnin_data`` and ``main_data``. The former is only used to
initialize the network the the latter is used to train the network. This
data process can be implemented by the following code:

.. code:: python

   data['burnin_obs'] = data['obs'][:bs]
   data['main_obs'] = data['obs'][bs:bs + self._nstep]
   data['target_obs'] = data['obs'][bs + self._nstep:]

.. note::
   Burn-in is not conflict with RNN reset. Use burn-in also needs RNN to reset by last timestep's hidden state. Burn-in only make a specific number of forward steps before usual forward.

For more details of RNN and burn-in, you can refer to `ding/policy/r2d2.py`.
