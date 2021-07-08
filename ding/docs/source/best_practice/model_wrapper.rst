How to Customize Model Wrapper
===============================

Function of Model Wrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^

Usually the output of a reinforcement learning model is value, Q or
action logits. DI-engine also requires customized model to output part 
or all of them. To improve the usability of model and support more 
functions, DI-engine provides model wrapper to sample action by specific 
strategy, adapt to RNN and other functions.

DI-engine provides the following model wrapper:

- BaseModelWrapper. Add reset method for model. In DI-engine's policy implementations, 
  many policy will call model's reset method in case of some method may use it 
  (e.g. HiddenStateWrapper). Wrap each model inherited from `nn.Module` using `model_wrap` 
  function will be wrapped by BaseModelWrapper automatically.

- HiddenStateWrapper. Add support for models which need to maintain
  hidden state like LSTM.

- SampleWrapper, including ArgmaxSampleWrapper,
  MultinomialSampleWrapper, EpsGreedySampleWrapper. Allow users to sample 
  action by argmax, multinomial or epsilon greedy strategy.

- ActionNoiseWrapper. To add noise on output action, mostly used in
  continuous action space environment.

- TargetNetworkWrapper. Add a target network for base model, used in DQN
  and many other RL algorithms.

Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users should follow the following steps to customize a model wrapper:

1. Define your model wrapper class like other wrappers in
   ``ding/model/wrappers/model_wrappers.py``;

2. Add your wrapper's name into ``ding/model/wrappers/model_wrappers.py: wrapper_name_map`` or use wrapper
   register to make sure your wrapper can be retrieved by ding. In the former
   way you can directly use your model wrapper by name without extra registration
   or you may need to regist it. Usually the format is like this:

.. code:: python

   @WRAPPER_REGISTRY.register('your_wrapper_name')
   class YourWrapper('IModelWrapper'):

       pass
     

3. Wrap your model with `model_wrap` function.

.. code:: python

  wrapped_model = model_wrap(origin_model, wrapper_name='your_wrapper_name', **kwargs)

.. note::
   All model wrappers **must** inherit from ``IModelWrapper``.

We show the implementation of HiddenStateWrapper in DI-engine to explain
how to customize a model wrapper.

If we want to use RNN in our model, we'll have to maintian hidden states
during the training process. With HiddenStateWrapper we can achieve this 
goal without changing code of policy.

The structure of HiddenStateWrapper are as follows:

.. code:: python

   class HiddenStateWrapper(IModelWrapper):

       def __init__(
          self, model: Any, state_num: int, save_prev_state: bool = False, init_fn: Callable = lambda: None
          ) -> None:

       """
    
       Overview:
    
         Maintain the hidden state for RNN-base model. Each sample in a batch has its own state. 
         Init the maintain state and state function; Then wrap the ``model.forward`` method with auto 
         saved data ['prev_state'] input, and create the ``model.reset`` method.
  
       Arguments:
    
         - model(:obj:`Any`): Wrapped model class, should contain forward method.
         - state_num (:obj:`int`): Number of states to process.
         - save_prev_state (:obj:`bool`): Whether to output the prev state in output['prev_state'].   
         - init_fn (:obj:`Callable`): The function which is used to init every hidden state when init and reset. 
           Default return None for hidden states.
    
       """

         ...

     def forward(self, data, **kwargs):
         ...
         return output

     def reset(self, *args, **kwargs):
         ...

     def reset_state(self, state: Optional[list] = None, state_id: Optional[list] = None) -> None:
         ...

     def before_forward(self, data: dict, state_id: Optional[list]) -> Tuple[dict, dict]:
         ...

     def after_forward(self, h: Any, state_info: dict, valid_id: Optional[list] = None) -> None:
         ...

- ``__init__``: Initialize hidden state as arguments, save it as model
  property ``self._state``

- ``before_forward``: Put ``self._state`` into model input data, the key
  is 'prev_state'

- ``after_forward``: Save model's output ``next_state`` into
  ``self._state``

- ``reset``: Reset wrapper related state, e.g. hidden state in RNN

- ``forward``: Call ``before_forward``, ``forward`` function of model,
  ``after_forward`` in turn

The dataflow of this process is as follows:

        .. image:: images/model_hiddenwrapper_img.png
            :align: center
            :scale: 60%

Other examples of model wrapper can be found in
``ding/model/wrappers/model_wrappers.py``, you can find more details
there.
