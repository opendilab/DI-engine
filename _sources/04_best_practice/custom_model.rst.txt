How to customize the neural network model
=================================================

In using reinforcement learning methods, one would have to choose an appropriate neural networks depending on the nature of the decision problem and the policy that is used. In the context of the DI-engine framework, a user can do so in 2 primary ways. The first way involves a user making use of the configuration file ``cfg.policy.model`` to automatically generate the desired neural network. The second way gives the user more control by allowing the desired neural network (instantiated as an object) to be passed direcly into the policy.

The purpose of this guide is to explain the details with regards to these 2 primary ways of choosing the appropriate neural network and as well as the principles behind them. 

Default model used in a policy 
----------------------------------

For a policy implemented in DI-engine, the ``default_model`` method contains the details of the default neural network model that was implemented. Take for example the SACPolicy implementation:

.. code:: python

   @POLICY_REGISTRY.register('sac')
    class SACPolicy(Policy):
    ...

        def default_model(self) -> Tuple[str, List[str]]:
            if self._cfg.multi_agent:
                return 'maqac_continuous', ['ding.model.template.maqac']
            else:
                return 'qac', ['ding.model.template.qac']
    ...

Observe here that the method either returns \ ``'maqac_continuous', ['ding.model.template.maqac']``\  or \ ``'qac', ['ding.model.template.qac']``\. In both cases, the first item in the return tuple is the name registered with DI-engine's model registry mechanism. The second item gives an indication of the file path of where the model file is located.

When using the configuration file ``cfg.policy.model``, DI-engine will correspondingly pass each argument into the model registered with DI-engine's registry mechanism. (For example, argument ``obs_shape``, ``action_shape`` etc will be passed into `QAC <https://github.com/opendilab/DI-engine/blob/main/ding/model/template/qac.py#L13>`_ ). The required neural network is then automatically generated in the model class based on the incoming arguments (e.g. a fully connected layer (FC) for vector input and a convolution (Conv) for image input).

How to customize the neural network model
-------------------------------------------

It is often the case that the \ ``default_model``\ chosen in a DI-engine \ ``policy``\  is not suitable for one's task at hand. Take for example the use of \ ``sac``\  on the \ ``cartpole-swingup``\  task of \ ``dmc2gym``\  (a wrapper for the Deep Mind Control Suite). Note the default values for observation is  \ ``pixel``\, while \ ``obs_shape = (3, height, width)``\  (For setting \ ``from_pixel = True, channels_first = True``\, see \ `dmc2gym <https://github.com/opendilab/DI-engine-docs/blob/main/source/13_envs/dmc2gym_zh.rst>`__\  documentation for details)

If one were to look at the source code of \ `sac <https://github.com/opendilab/DI-engine/blob/main/ding/policy/sac.py>`__\, it can be seen that the \ ``default_model``\  is actually \ `qac <https://github.com/opendilab/DI-engine/blob/main/ding/model/template/qac.py>`__\. The \ ``qac model``\  currently only supports an \ ``obs_shape``\  of one dimensoin (e.g. (4, )). Hence, it becomes apparent that one must customize a model according to one's needs and ensure that the policy is setup accordingly.

Step-by-step guide to customizing a model
------------------------------------------

1. Choose your environment and policy
+++++++++++++++++++++++++++++++++++++

-  For the purpose of this guide, let the choice of environment and policy to be the use of \ ``sac``\  on the \ ``cartpole-swingup``\  task of \ ``dmc2gym``\  (a wrapper for the Deep Mind Control Suite). (For details, see \ `dmc2gym <https://github.com/opendilab/DI-engine-docs/blob/main/source/13_envs/dmc2gym_zh.rst>`__\  documentation)

2. Check to see if the policy's default_model is suitable
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

-  This can be done in 1 of 2 ways. One either look up the documentation at \ `policy-default_model <https://xxx>`__\  or read the source code of \ `ding/policy/sac:SACPolicy <https://github.com/opendilab/DI-engine/blob/main/ding/policy/sac.py>`__\  and find out what is being used in the \ ``default_model``\  method. 

.. code:: python

   @POLICY_REGISTRY.register('sac')
    class SACPolicy(Policy):
    ...

        def default_model(self) -> Tuple[str, List[str]]:
            if self._cfg.multi_agent:
                return 'maqac_continuous', ['ding.model.template.maqac']
            else:
                return 'qac', ['ding.model.template.qac']
    ...

-  Now that we see QAC is being used here, we can then further read up \ `ding/model/template/qac:QAC <https://github.com/opendilab/DI-engine/blob/69db77e2e54a0fba95d83c9411c6b11cd25beae9/ding/model/template/qac.py#L40>`__\. The \ ``qac model``\  implemented in DI-engine currently only supports \ ``obs_shape``\ of 1. However, the observation space of the task chosen is an image of \ ``obs_shape = (3, height, width)``\

Hence, we will need to do some customization.

3. Customizing the model
+++++++++++++++++++++++++++++++++++++

Using the default_model as a guide and reference when crafting the custom_model:

-  All public methods in the default_model must be implemented in custom_model.

-  Ensure that the type of return in custom_model is the same as the default_model.

One can also reference the \ ``encoder``\  implementation of \ ``encoder.py``\  and \ ``head``\  implementation of \ ``head.py``\. See \ `ding/model/common <https://github.com/opendilab/DI-engine/tree/main/ding/model/common>`__\

-   The \ ``encoder``\  is used to encode inputs such as \ ``obs``\ , \ ``action``\  etc. for subsequent processing. DI-engine have thus far implemented the following encoders:

+-----------------------+-------------------------------------+
|encoder                |usage                                |
+=======================+=====================================+
|ConvEncoder            |For encoding image inputs            |
+-----------------------+-------------------------------------+
|FCEncoder              |For encoding one dimensional inputs  |                
+-----------------------+-------------------------------------+
|StructEncoder          |                                     |
+-----------------------+-------------------------------------+

-  The \ ``head``\  is used to process the encoded inputs and outputs data required by the policy or the overall RL process. DI-engine have thus far implemented the following heads:

+-----------------------+-------------------------------------+
|head                   |usage                                |
+=======================+=====================================+
|DiscreteHead           |Output discrete action value         |
+-----------------------+-------------------------------------+
|DistributionHead       |Output Q value distribution          |
+-----------------------+-------------------------------------+
|RainbowHead            |Output Q value distribution          |
+-----------------------+-------------------------------------+
|QRDQNHead              |Quantile regression                  |
|                       |continuous action value              |
+-----------------------+-------------------------------------+
|QuantileHead           |Output action quantiles              |
+-----------------------+-------------------------------------+
|DuelingHead            |Output discrete action value logits  |
+-----------------------+-------------------------------------+
|RegressionHead         |Output continuous action Q values    |
+-----------------------+-------------------------------------+
|ReparameterizationHead |Output action mu and sigma           |
+-----------------------+-------------------------------------+
|MultiHead              |Multi-dimensional action spaces      |
+-----------------------+-------------------------------------+

From here, one will customize the model required specifically for the sac+dmc2gym+cartpole-swingup task combination. For now, we will name and instantiate the new custom_model as a \ ``QACPixel``\  type.

-  With reference to the \ ``QAC``\  implementation, the \ ``QACPixel``\  implementation must have the following methods:  \ ``init``\, \ ``forward``\, \ ``compute_actor``\  and \ ``compute_critic``\.

.. code:: python

  @MODEL_REGISTRY.register('qac')
    class QAC(nn.Module):
    ...
      def __init__(self, ...) -> None:
        ...
      def forward(self, ...) -> Dict[str, torch.Tensor]:
        ...
      def compute_actor(self, obs: torch.Tensor) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        ...
      def compute_critic(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ...

-  In the case of image inputs, the \ ``init``\ method of \ ``QACPixel``\  will have to adjust the definition its \ ``self.actor``\  and \ ``self.critic``\. By observation, we can see that the \ ``self.action``\  and \ ``self.critic``\  of \ ``QAC``\  uses an encoder that consists of only a single layer nn.Linear.

.. code:: python

  @MODEL_REGISTRY.register('qac')
  class QAC(nn.Module):
  ...
    def __init__(self, ...) -> None:
      ...
      self.actor = nn.Sequential(
              nn.Linear(obs_shape, actor_head_hidden_size), activation,
              ReparameterizationHead(
                  ...
              )
          )
      ...
      self.critic = nn.Sequential(
              nn.Linear(critic_input_size, critic_head_hidden_size), activation,
              RegressionHead(
                  ...
              )
          )

-  We define the type of encoder by defining the variable encoder_cls. In this case, we have defined it as a \ ``ConvEncoder``\. Since we need to connect the encoded obs with the action, \ ``self.critic``\  is constructed from 2 parts: one part being \ ``self.critic_encoder``\  and the other part \ ``self.critic_head``\.

.. code:: python

  @MODEL_REGISTRY.register('qac_pixel')
  class QACPixel(nn.Module):
  def __init__(self, ...) -> None:
      ...
      encoder_cls = ConvEncoder
      ...
      self.actor = nn.Sequential(
            encoder_cls(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type),
            ReparameterizationHead(
                ...
            )
        )
      ...
      self.critic_encoder = global_encoder_cls(obs_shape, encoder_hidden_size_list, activation=activation,
                                                     norm_type=norm_type)
      self.critic_head = RegressionHead(
          ...
      )
      self.critic = nn.ModuleList([self.critic_encoder, self.critic_head])

-  Finally, we will also have to make corresponding changes to \ ``compute_actor``\  and  \ ``compute_critic``\

4. How to make use of a customized model
++++++++++++++++++++++++++++++++++++++++++

-  New pipeline: Define the model with the corresponding imports, then pass the model into the policy as an argument as follows.

.. code:: python
   
   ...
   from ding.model.template.qac import QACPixel
   ...
   model = QACPixel(**cfg.policy.model)
   policy = SACPolicy(cfg.policy, model=model) 
   ...


-  Old pipeline: Pass the defined model into \ `serial_pipeline <https://github.com/opendilab/DI-engine/blob/main/ding/entry/serial_entry.py#L22>`__\  as a argument. The model will then be passed on to \ ``create_policy``\. 

.. code:: python
  
  ...
  def serial_pipeline(
    input_cfg: Union[str, Tuple[dict, dict]],
    seed: int = 0,
    env_setting: Optional[List[Any]] = None,
    model: Optional[torch.nn.Module] = None,
    max_train_iter: Optional[int] = int(1e10),
    max_env_step: Optional[int] = int(1e10),
    ) -> 'Policy':
    ...
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])
    ...

5. Unit testing a customized model
+++++++++++++++++++++++++++++++++++++

-  In general, when writing unit tests, one would need to first manually construct the \ ``obs``\  and \ ``action``\  inputs, define the model and verify that output dimensions and type are correct. Following that, if the model contains a neural network, it is also necessary to verify that the model is differentiable.

Take for example the unit test written for our new model \ ``QACPixel``\. We first construct \ ``obs``\ with a shape of  \ ``(B, channel, height, width)``\  (where B = batch_size) and we construct \ ``action``\  with a shape of \ ``(B, action_shape)``\. Then we define the model \ ``QACPixel``\  and obtain and pass along the corresponding outputs of its \ ``actor``\  and \ ``critic``\. Finally, we make sure that the shape sizes of \ ``q, mu, sigma``\  are correct and that \ ``actor``\  and \ ``critic``\  is differentiable.

.. code:: python

  class TestQACPiexl:

    def test_qacpixel(self, action_shape, twin):
      inputs = {'obs': torch.randn(B, 3, 100, 100), 'action': torch.randn(B, squeeze(action_shape))}
      model = QACPixel(
          obs_shape=(3,100,100 ),
          action_shape=action_shape,
          ...
      )
      ...
      q = model(inputs, mode='compute_critic')['q_value']
      if twin:
          is_differentiable(q[0].sum(), model.critic[0])
          is_differentiable(q[1].sum(), model.critic[1])
      else:
          is_differentiable(q.sum(), model.critic_head)

      (mu, sigma) = model(inputs['obs'], mode='compute_actor')['logit']
      assert mu.shape == (B, *action_shape)
      assert sigma.shape == (B, *action_shape)
      is_differentiable(mu.sum() + sigma.sum(), model.actor)

.. tip::

  Alternatively, users can also reference existing unit tests implemented in DI-engine to get familiar with the various neural networks while customizing a model.

 For more on writing and running unit tests, refer to \ `Unit Testing Guidelines <https://di-engine-docs.readthedocs.io/zh_CN/latest/22_test/index_zh.html>`__\ 
