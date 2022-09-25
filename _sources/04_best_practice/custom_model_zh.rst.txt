如何自定义神经网络模型（model）
=================================================

Policy 默认使用的模型是什么
----------------------------------

DI-engine 中已经实现的 policy，默认使用 default_model 方法中表明的神经网络模型，例如在 SACPolicy 中：

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

此处return的 \ ``'maqac_continuous', ['ding.model.template.maqac']``\ ，前者是模型在注册器中注册的名字，后者是模型所处的文件路径。

如何自定义神经网络模型
----------------------------------

但很多时候 DI-engine 中实现的 \ ``policy``\ 中的  \ ``default_model``\ 不适用自己的任务，例如这里想要在 \ ``dmc2gym``\ 环境 \ ``cartpole-swingup``\  任务下应用 \ ``sac``\ 算法，且环境 observation 为  \ ``pixel``\ ，
即 \ ``obs_shape = (3, height, width)``\ （如果设置 \ ``from_pixel = True, channels_first = True``\ ，详情见  \ `dmc2gym 环境文档 <https://github.com/opendilab/DI-engine-docs/blob/main/source/13_envs/dmc2gym_zh.rst>`__\ ） 

而此时查阅 \ `sac 源码 <https://github.com/opendilab/DI-engine/blob/main/ding/policy/sac.py>`__\ 可知 \ ``default_model``\ 为 \ `qac <https://github.com/opendilab/DI-engine/blob/main/ding/model/template/qac.py>`__\ ，
\ ``qac model``\ 中暂时只支持 \ ``obs_shape``\ 为一维的情况，此时我们即可根据需求自定义 model 并应用到 policy。

自定义 model 基本步骤
----------------------------------

1. 明确 env, policy
+++++++++++++++++++++++++++++++++++++

-  比如这里选定 \ ``dmc2gym``\ 环境 \ ``cartpole-swingup``\  任务，且设置 \ ``from_pixel = True, channels_first = True``\ （详情见  \ `dmc2gym 环境文档 <https://github.com/opendilab/DI-engine-docs/blob/main/source/13_envs/dmc2gym_zh.rst>`__\ ） 
   ，即此时观察空间为图像 \ ``obs_shape = (3, height, width)``\ ，并选择 \ ``sac``\ 算法进行学习。


2. 查阅 policy 中的 default_model 是否适用
++++++++++++++++++++++++++++++++++++++++++

-  此时根据\ `policy-default_model 链接 <https://xxx>`__\ 或者直接查阅源码 \ `ding/policy/sac:SACPolicy <https://github.com/opendilab/DI-engine/blob/main/ding/policy/sac.py>`__\ ，找到 SAC 的  \ ``default_model``\：

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

-  进一步查看  \ `ding/model/template/qac:QAC <https://github.com/opendilab/DI-engine/blob/69db77e2e54a0fba95d83c9411c6b11cd25beae9/ding/model/template/qac.py#L40>`__\ ，
   发现 DI-engine 中实现的 \ ``qac model``\ 暂时只支持 \ ``obs_shape``\ 为一维的情况，但是此时环境的观察空间为图像 \ ``obs_shape = (3, height, width)``\ ，
   因此我们需要根据需求自定义 model 并应用到 policy。

3. custom_model 实现
+++++++++++++++++++++++++++++++++++++

根据已有的 defaul_model 来决定 custom_model 所需实现的功能:

-  需要实现原 default model 中所有的 public 方法
  
-  保证返回值的类型的原 default model 一致

具体实现可利用 \ `ding/model/common <https://github.com/opendilab/DI-engine/tree/main/ding/model/common>`__\ 下 \ ``encoder.py``\ / \ ``head.py``\ 已实现的 \ ``encoder``\ 和 \ ``head``\ 

-  \ ``encoder``\ 用于对输入的 \ ``obs``\ 或者 \ ``action``\ 等进行编码，便于进行后续处理， DI-engine 中已实现的 encoder 如下：

+-----------------------+-------------------------------------+
|encoder                |usage                                |
+=======================+=====================================+
|ConvEncoder            |处理图像obs输入                      |
+-----------------------+-------------------------------------+
|FCEncoder              |处理一维obs输入                      |                
+-----------------------+-------------------------------------+
|StructEncoder          |                                     |
+-----------------------+-------------------------------------+

-  \ ``head``\ 用于对已经编码的数据进行相应处理，输出 policy 所需信息或者辅助 RL 过程， DI-engine 中已实现的 head ：

+-----------------------+-------------------------------------+
|head                   |usage                                |
+=======================+=====================================+
|DiscreteHead           |输出离散动作值                       |
+-----------------------+-------------------------------------+
|DistributionHead       |输出 Q 值分布                        |
+-----------------------+-------------------------------------+
|RainbowHead            |输出 Q 值分布                        |
+-----------------------+-------------------------------------+
|QRDQNHead              |Quantile Regression DQN，            |
|                       |用于输出动作分位数                   |
+-----------------------+-------------------------------------+
|QuantileHead           |用于输出动作分位数                   |
+-----------------------+-------------------------------------+
|DuelingHead            |用于输出离散动作的 logit             |
+-----------------------+-------------------------------------+
|RegressionHead         |用于输出动作 Q 值                    |
+-----------------------+-------------------------------------+
|ReparameterizationHead |用于输出动作 mu 和 sigma             |
+-----------------------+-------------------------------------+
|MultiHead              |处理动作空间为多维的情况             |
+-----------------------+-------------------------------------+


例如这里需要自定义针对 sac+dmc2gym+cartpole-swingup 任务的 model ，我们把新的 custom_model 实现为 \ ``QACPixel``\  类

-  这里对照 \ ``QAC``\ 已经实现的方法， \ ``QACPixel``\ 需要实现 \ ``init``\  ， \ ``forward``\ ，以及 \ ``compute_actor``\ 和  \ ``compute_critic``\ 。

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

-  针对图像输入， \ ``QACPixel``\ 主要需要修改的是 \ ``init``\ 中对 \ ``self.actor``\ 和 \ ``self.critic``\ 的定义。
   可以看到 \ ``QAC``\ 中 \ ``self.actor``\ 和 \ ``self.critic``\ 的 encoder 都只是一层 nn.Linear

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

-  我们通过定义 encoder_cls 指定 encoder 的类型，加入 \ ``ConvEncoder``\ ，并且因为需要对 obs 进行encode 后和 action 进行拼接，
   将 \ ``self.critic``\ 分为  \ ``self.critic_encoder``\ 和 \ ``self.critic_head``\ 两部分

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

-  再对 \ ``compute_actor``\ 和  \ ``compute_critic``\ 分别进行修改即可。

4. 如何应用自定义模型
+++++++++++++++++++++++++++++++++++++

-  新 pipeline ： 直接定义model，作为参数传入 policy 进行初始化，如：

.. code:: python
   
   ...
   from ding.model.template.qac import QACPixel
   ...
   model = QACPixel(**cfg.policy.model)
   policy = SACPolicy(cfg.policy, model=model) 
   ...


-  旧pipeline

将定义好的 model 作为参数传入 \ `serial_pipeline <https://github.com/opendilab/DI-engine/blob/main/ding/entry/serial_entry.py#L22>`__\ , 
传入的 model 将在 \ `serial_pipeline <https://github.com/opendilab/DI-engine/blob/main/ding/entry/serial_entry.py#L59>`__\ 
通过 \ ``create_policy``\  被调用。或者跟上述新 pipeline 一样，作为参数传入 policy 。

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

5. 测试自定义 model 
+++++++++++++++++++++++++++++++++++++

-  编写新的 model 测试，一般而言，首先需要构造 \ ``obs``\  \ ``action``\ 等输入，传入 model ，验证输出的维度、类型的正确性。其次如果涉及神经网络，需要验证 model 是否可微。
   如对于我们编写的新模型 \ ``QACPixel``\ 编写测试，首先构造维度为 \ ``(B, channel, height, width)``\ （B = batch_size）的 \ ``obs``\ 和维度为 \ ``(B, action_shape)``\ 的 \ ``obs``\ ，传入 \ ``QACPixel``\ 的 \ ``actor``\ 和 \ ``critic``\ 得到输出.
   检查输出的 \ ``q, mu, sigma``\ 的维度是否正确，以及相应的 \ ``actor``\ 和 \ ``critic``\  model 是否可微：

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

  同样，使用者也可以参考 DI-engine 中已有的单元测试，来熟悉相关神经网络模型的使用

-  单元测试编写运行可参考 \ `单元测试指南 <https://di-engine-docs.readthedocs.io/zh_CN/latest/22_test/index_zh.html>`__\ 
