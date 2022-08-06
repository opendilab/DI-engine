如何自定义模型 Wrapper
=================================================

模型包装器的功能
^^^^^^^^^^^^^^^^^^^^^^^^^^

通常强化学习模型的输出是 V, Q 或者动作 logits。 DI-engine需要定制模型来输出这些的部分或者全部。 为了提高模型的可用性并支持更多
功能, DI-engine 提供模型 wrapper 去通过某个具体的策略（比如RNN或者其他方程）去采样动作。

DI-engine 提供以下模型 wrapper：

- BaseModelWrapper： 为模型添加重置方法。 在 DI-engine 的策略实现中，
  许多策略会调用模型的重置方法（例如 HiddenStateWrapper）。 任何继承了 `nn.Module` 的模型在使用 `model_wrap` 函数来 wrap 后将会自动被 ``BaseModelWrapper`` wrap。

- HiddenStateWrapper： 用于需要维护 hidden state 的模型，比如LSTM。
  
- SampleWrapper, 包括 ArgmaxSampleWrapper，
  MultinomialSampleWrapper，EpsGreedySampleWrapper：允许用户通过 argmax、多项式分布或 epsilon 贪心策略采样动作。

- ActionNoiseWrapper：在输出动作上添加噪声，主要用于连续动作空间环境。

- TargetNetworkWrapper：为基本模型添加目标网络相关功能，用于需要目标网络的 DQN 等 RL 算法。



举例
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

用户可以参考以下步骤自定义模型 wrapper：

1. 像其他 wrapper 一样定义模型 wrapper 类
   ``ding/model/wrappers/model_wrappers.py``;

2. 将 wrapper 的名称添加到 ding/model/wrappers/model_wrappers.py: wrapper_name_map 或使用 wrapper
   注册以确保可以通过 ding 检索您的 wrapper。前者的话， 无需额外注册即可通过 wrapper_name 指定调用的模型 wrapper，
   后者的话，您需要注册这个 wrapper 。通常格式如下：

.. code:: python

   @WRAPPER_REGISTRY.register('your_wrapper_name')
   class YourWrapper('IModelWrapper'):

       pass
     

3. 调用 `model_wrap` 函数包装你的模型。

.. code:: python

  wrapped_model = model_wrap(origin_model, wrapper_name='your_wrapper_name', **kwargs)

.. note::
   所有 model wrapper **必须** 继承 ``IModelWrapper``。

我们将在下面展示 DI-engine 中 HiddenStateWrapper 的实现，以用来解释如何自定义模型 wrapper。

如果我们想在我们的模型中使用 RNN，我们必须在训练过程中保存 hidden state。通过使用 Hidden StateWrapper， 我们可以在不更改政策代码的情况下实现这一点。

HiddenStateWrapper 的结构如下：

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

这个过程的数据流如下：

        .. image:: images/model_hiddenwrapper_img.png
            :align: center
            :scale: 60%

关于模型 wrapper 的其他示例，您可以在 ``ding/model/wrappers/model_wrappers.py`` 找到更多细节。
