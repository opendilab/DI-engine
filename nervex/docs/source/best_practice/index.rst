Best Practice
~~~~~~~~~~~~~~~

如何给算法添加priority
======================

1. config设置\ ``use_priority``

   .. code:: python

      policy=dict(
          ...,
          use_priority=True,
          ...,
      )

2. 使用priority

   -  replay
      buffer会默认根据priority进行采样（\ ``use_priority=False``\ ，replay
      buffer内部\ ``priority=1.0``\ ）

   -  replay
      buffer会给数据dict添加一项数据\ ``IS``\ ，需要将其乘在loss计算的最后（每个样本一个值）

      .. code:: python

         import torch.nn.functional as F

         # output: (B, ), target: (B, )
         # not use IS
         loss = F.mse_loss(output, target)
         # use IS
         loss = (F.mse_loss(output, target, reduction='none') * data['IS']).mean()
         # nervex td error(data['weight'] = data['IS'], assigned in policy._data_preprocess_learn method)
         data_n = q_nstep_td_data(
           q_value, target_q_value, data['action'], target_q_action, reward, data['done'], data['weight']
         )
         loss, td_error_per_sample = q_nstep_td_error(data_n, self._gamma, nstep=self._nstep)

3. 更新priority

   将要更新的\ ``priority``\ 值作为\ ``_forward_learn``\ 方法返回值的一个键值对，键必须是\ ``priority``\ ，值必须是\ ``list``\ ，且长度等于batch_size

   .. code:: python

      loss, td_error_per_sample = q_nstep_td_error(data_n, self._gamma, nstep=self._nstep)
      return {
          'total_loss': loss.item(),
          'priority': td_error_per_sample.abs().tolist(),
      }

4. A-pex相关

   -  ``policy``\ 的\ ``_forward_collect``\ 方法中也计算priority，并作为键值对返回出去

   -  ``policy``\ 的\ ``_process_transition``\ 方法中将\ ``armor_output``\ 中的\ ``priority``\ 放入返回数据中，加入buffer后会自动作为初始值

      .. code:: python

         def _process_transition(self, obs: Any, armor_output: dict, timestep: namedtuple) -> dict:
         		transition = {
                 'obs': obs,
                 'next_obs': timestep.obs,
                 'action': armor_output['action'],
                 'priority': armor_output['priority'],
                 'reward': timestep.reward，
                 'done': timestep.done,
             }
             return EasyDict(transition)

   -  对于A-pex中的多个Actor使用不同的探索策略，目前仅支持在nervex
      parallel入口实现，需要给每一个actor task设定相关参数

5. 更改priority采样的相关参数

   .. code:: python

      replay_buffer=dict(
          buffer_name=['agent'],
          agent=dict(
              meta_maxlen=4096,
              max_reuse=16,
              alpha=0.6,
              beta=0.4,
              # sample step count
              anneal_step=0,
          )
      )

.. _header-n21:

使用 multi discrete action space
=======================================

1. 环境空间定义

   .. code:: python

      # 3 crossing
      # action dim: {'htxdj_wjj': 2, 'haxl_wjj': 2, 'haxl_htxdj': 3}
      logit_shape = [torch.Size([4, 2]), torch.Size([4, 2]), torch.Size([4, 3])]
      action_shape = [torch.Size([4]), torch.Size([4]), torch.Size([4])]

2. 中间数据处理操作

   -  多个环境的数据合成batch来inference

   -  合成batch inference之后拆分对应每个环境，组装成一个个transition

   -  训练前多个样本组装成一个batch

      .. code:: python

         from nervex.data import default_collate, default_decollate

         # list, dict, tuple, scalar, np.ndarray, torch.Tensor
         split_data = [torch.randn(4) for _ in range(8)]
         batch = default_collate(split_data)
         assert batch.shape == (8, 4)
         split_data = default_decollate(batch)

3. forward_learn（多次调用，每次是一个标准的计算过程）

   .. code:: python

      tl_num = len(q_value)
      loss = []
      for i in range(tl_num):
          td_data = q_1step_td_data(
              q_value[i], target_q_value[i], data['action'][i], next_act[i], data['reward'], data['done'],
              data['weight']
          )
          loss.append(q_1step_td_error(td_data, self._gamma))
          loss = sum(loss) / (len(loss) + 1e-8)

.. _header-n32:

RNN适配
==========

1. 隐状态维护

   使用\ ``HiddenStatePlugin``\ 来进行维护

   .. code:: python

      from typing import Any
      from nervex.armor import Armor

      # create plugin
      model: torch.nn.Module
      batch_size = 8
      armor = Armor(model)
      armor.add_model('target', update_type='assign', update_kwargs={'freq': 500})
      armor.add_plugin('main', 'hidden_state', state_num=batch_size)
      armor.add_plugin('target', 'hidden_state', state_num=batch_size)

      # reset state
      init_state: Any
      armor.reset(data_id=None, state=init_state)
      output1 = armor.forward(inputs1)
      output2 = armor.forward(inputs2)
      # reset the state of sample0 with init_state[1]
      armor.reset(data_id=[0], state=init_state[1])
      output3 = armor.forward(inputs3)

2. actor->learner传递数据

   注册能够返回当前帧输入state的plugin

   .. code:: python

      from nervex.armor import Armor

      armor = Armor(model)
      # indicate save_prev_state=True
      armor.add_plugin('main', 'hidden_state', state_num=env_num, save_prev_state=True)
      init_state: Any
      armor.reset(data_id=None, state=init_state)
      output = armor.forward(inputs)
      prev_state = output['prev_state']
      assert isinstance(list, prev_state) and len(prev_state) == env_num

   同样在policy的\ ``_process_transition``\ 方法中添加prev_state即可

3. learner数据组装

   使用timestep_collate

   .. code:: python

      from nervex.data import timestep_collate

      timestep_batch = timestep_collate(data)
      # timestep_batch: (T, B, *)

4. burnin

   参考policy/r2d2.py的learn部分

.. _header-n318:

learner日志中添加变量
=====================

1. 傻瓜用法

2. 不定期出现的变量

3. 深度定制化用法

.. _header-n74:

定制化优化器
===============

1. 更换优化器

   ``nerveX`` 框架中 ，由 ``policy`` 类中的 ``_init_learn`` 方法进行优化器的初始化：
   
   .. code:: python

      def _init_learn(self) -> None:
      r"""
      Overview:
         Learn mode init method. Called by ``self.__init__``.
         Init optimizers, algorithm config, main and target armors.
      """
      # init optimizer
      self._optimizer = Adam(
         self._model.parameters(),
         lr=self._cfg.learn.learning_rate_actor,
         weight_decay=self._cfg.learn.weight_decay
      )
   
   如需对优化器进行更换，只需修改对应算法 ``policy`` 类 ``_init_learn`` 方法中的对应代码即可。

   此外，``nerveX`` 框架中对优化器进行了重写，在继承了 ``torch.optim`` 类的前提下实现了一些特定的梯度操作。
   具体代码可以参考 ``nervex\torch_utils\optimizer_helper.py`` 。 在实际使用时，可以根据需要直接使用 ``torch`` 自带优化器或重写后的优化器。

2. 多个优化器 or hook

   某些算法的神经网络可能由多个部分组成，如 ``DDPG`` 算法的网络就由 ``actor`` 和 ``critic`` 两部分构成。 
   
   在更新某一部分神经网络的参数时，可能需要另一部分网络的对应输出，但不希望另一部分的网络参数因此更新；如 ``DDPG`` 算法在更新 ``actor`` 部分的网络时loss需要根据 ``critic`` 进行计算，但不希望 ``cirtic`` 的梯度更新。

   此时，我们可以使用多个优化器，分别对神经网络的不同组成部分进行更新，以 ``DDPG`` 算法为例，在 ``_init_learn``  方法中初始化了多个优化器：

   .. code:: python

      def _init_learn(self) -> None:
      r"""
      Overview:
         Learn mode init method. Called by ``self.__init__``.
         Init actor and critic optimizers, algorithm config, main and target armors.
      """
      # actor and critic optimizer
      self._optimizer_actor = Adam(
         self._model.actor.parameters(),
         lr=self._cfg.learn.learning_rate_actor,
         weight_decay=self._cfg.learn.weight_decay
      )
      self._optimizer_critic = Adam(
         self._model.critic.parameters(),
         lr=self._cfg.learn.learning_rate_critic,
         weight_decay=self._cfg.learn.weight_decay
      )
      # ...

   
   在 ``_forward_learn`` 时 ``actor`` 和 ``critic`` 分别根据对应loss和优化器进行更新。

   此外，对 ``torch`` 机制更熟悉的使用者可以使用在神经网络中设定梯度相关的 ``backward_hook`` 的方式对神经网络的不同组成部分进行更新，但此种方式的实现逻辑相对复杂，对使用者的要求更高。


3. grad clip/ignore
   
   许多算法/论文中，都对某些情况下的梯度进行了裁剪或忽略操作，即 ``grad_clip`` 和 ``grad_ignore`` 操作。
   这些操作自然可以在 ``_forward_learn`` 方法中调用对应函数完成，如：

   .. code:: python

      from torch.nn.utils import clip_grad_norm

      # ...

      optimizer.zero_grad()
      loss.backward()
      clip_grad_norm(model.parameters(), clip_value)
      optimizer.step()
   
   但有些相对复杂的梯度操作需要用到优化器中的梯度相关信息，因此较为方便的实现方式即为在优化器内实现对应的梯度操作。
   为此，我们在 ``nervex\torch_utils\optimizer_helper.py`` 中根据梯度操作对优化器进行了重写，方便使用。
   如需在 ``Adam`` 优化器中使用最简单的梯度裁剪时，只需在初始化时对grad_clip操作进行定义：

   .. code:: python

      from nervex.torch_utils.optimizer_helper import Adam

      # ...
      self._optimizer = Adam(
         self._model.parameters(),
         lr=self._cfg.learn.learning_rate_actor,
         weight_decay=self._cfg.learn.weight_decay,
         grad_clip_type='clip_value',
         clip_value=clip_value,
      )
   
   之后在进行 optimizer.step() 操作时会自动对梯度进行裁剪。

   除了简单的按数值进行梯度裁剪/忽略外，重写后的优化器还支持其他梯度操作，具体可以查看源码 ``nervex\torch_utils\optimizer_helper.py`` 或参考 ``nervex\torch_utils\tests\test_optimizer.py`` 测试文件中的使用方式。
