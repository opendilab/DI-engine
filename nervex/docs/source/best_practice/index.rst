Best Practice
~~~~~~~~~~~~~~~

1. 如何给算法添加priority
=========================

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

2. multi discrete action space
==============================

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

3. RNN适配
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

4. learner日志中添加变量
=========================

1. 傻瓜用法

   ``Policy`` 抽象基类中会默认包含两个变量 ``['cur_lr', 'total_loss']``，若有其他需要打印的变量，只需要修改两处：
   
   1. 在 ``Policy`` 的 ``_monitor_vars_learn`` 方法的返回值中额外加入 **变量名** 。如 PPO ：

      .. code:: python

         def _monitor_vars_learn(self) -> List[str]:
            return super()._monitor_vars_learn() + [
                  'policy_loss', 'value_loss', 'entropy_loss', 'adv_abs_max', 'approx_kl', 'clipfrac'
            ]
      
   2. 在 ``Policy`` 的 ``_forward_learn`` 方法的返回值中，以 dict 的形式返回 {变量名: 变量值} **键值对** 。如 PPO ：

      .. code:: python
         
         def _forward_learn(self, data: dict) -> Dict[str, Any]:

            # ...
            # ====================
            # PPO update
            # ====================
            # ...
            return {
                  'cur_lr': self._optimizer.defaults['lr'],
                  'total_loss': total_loss.item(),
                  'policy_loss': ppo_loss.policy_loss.item(),
                  'value_loss': ppo_loss.value_loss.item(),
                  'entropy_loss': ppo_loss.entropy_loss.item(),
                  'adv_abs_max': adv.abs().max().item(),
                  'approx_kl': ppo_info.approx_kl,
                  'clipfrac': ppo_info.clipfrac,
            }
      
   .. note::

      在 nerveX 中，使用 ``LoggedModel`` 模块对变量进行追踪，保存一定时间滑动窗口内的值，
      并在窗口内进行一定操作（主要为各种统计量，如取平均值做平滑操作，取最大最小值等）。
      （有兴趣的可以具体查看其 `文档 <../feature/autolog_overview.html>`_ 了解更多内容）

      这些操作中，最常见、常用的就是 **取平均** ，我们也对所有 scalar 类型的变量（指 int, float 等标量）
      默认进行了取平均的操作，并打印在 tensorboard logger 中。
      （为了保持终端与 log 文件的简洁性，在 text logger 中，我们没有打印平均值，而只打印了瞬时值。）

2. 不定期出现的变量

   不定期出现的变量，指并非 policy 的每次 forward 都会返回的变量，它们可能每隔 n 个 iteration 才会计算并返回一次。
   其在使用上和傻瓜用法 **没有任何区别** ，都需要在 ``_monitor_vars_learn`` 中声明，并在需要的时候在 ``_forward_learn`` 中返回。

   但由于 nerveX 中 ``LoggedModel`` 是固定步长的滑动窗口，就会导致不定期出现的变量的窗口内操作，和其他定期出现的变量间存在 **微小的差异** 。
   例如 PPG：

      .. code:: python

         def _forward_learn(self, data: dict) -> Dict[str, Any]:

            # ...
            # =============
            # PPG update
            # =============
            # ...
            if self._train_step % self._cfg.learn.algo.aux_freq == 0:
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
      
      PPG 中 ``['aux_value_loss', 'auxiliary_loss', 'behavioral_cloning_loss']`` 这三个变量，
      每 ``self._cfg.learn.algo.aux_freq`` 次 forward 才会返回一次（为了方便，假定为每 5 次吧）。
      其他变量，如 ``'total_loss'`` 每次 forward 都会返回。
      
      ``LoggedModel`` 会在每次 forward 后递进一个时间步，但其时间窗口是固定长度的，假设为 10 次时间步。
      这就导致 ``'total_loss'`` 的取平均，是对 10 次 forward 的返回值取平均；
      而 ``'aux_value_loss'`` 的取平均，仅对 10 / 5 = 2 次 forward 的返回值取平均。

3. 深度定制化用法

   1. ``LoggedModel`` 统计量定制

      上文讲到 ``LoggedModel`` 默认对 scalar 类型的变量进行取平均的操作，如果需要进行其他类型的操作，
      可以参考 `buffer <../api_doc/data/structure.html#buffer>`_  中的 ``OutTickMonitor``，
      修改 `base learner <../api_doc/worker/learner/learner.html#base-learner>`_ 中的 ``TickMonitor``。
      主要注意 ``__register`` 方法中 ``__max_func`` 这类函数的实现，并记得注册 attribute（如 ``priority``） 的 property（如 ``max`` ``min``）。

      .. code:: python

         class OutTickMonitor(LoggedModel):
            out_time = LoggedValue(float)
            priority = LoggedValue(float)
            # ...

            def __init__(self, time_: 'BaseTime', expire: Union[int, float]):  # noqa
               LoggedModel.__init__(self, time_, expire)
               self.__register()

            def __register(self):

               def __avg_func(prop_name: str) -> float:
                     records = self.range_values[prop_name]()
                     _list = [_value for (_begin_time, _end_time), _value in records]
                     return sum(_list) / len(_list)

               def __max_func(prop_name: str) -> Union[float, int]:
                     records = self.range_values[prop_name]()
                     _list = [_value for (_begin_time, _end_time), _value in records]
                     return max(_list)

               def __min_func(prop_name: str) -> Union[float, int]:
                     records = self.range_values[prop_name]()
                     _list = [_value for (_begin_time, _end_time), _value in records]
                     return min(_list)

               self.register_attribute_value('avg', 'out_time', partial(__avg_func, prop_name='out_time'))
               self.register_attribute_value('avg', 'priority', partial(__avg_func, prop_name='priority'))
               self.register_attribute_value('max', 'priority', partial(__max_func, prop_name='priority'))
               self.register_attribute_value('min', 'priority', partial(__min_func, prop_name='priority'))
               # ...

   2. Scalar类型之外的变量（如Histogram）

      对于要在 tensorboard logger 中打印的变量，我们都默认为 Scalar 类型，若有其他类型的打印需求，
      需要在 ``Policy`` 的 ``_forward_learn`` 方法的返回值中特别标明。
      
      例如，针对离散的动作，我想打印一个 batch 中的分布情况，需要修改的地方为：

      .. code:: python

         def _forward_learn(self, data: dict) -> Dict[str, Any]:

            # ...
            # =============
            # after update
            # =============
            # ...
            return {
                  # ...
                  '[histogram]action_distribution': data['action'],
            }

      dict 中键的命名方式为 ``'[VAR-TYPE]VAR-NAME'``，用 ``'[]'`` 来标示变量类型。

      .. note::

         由于 learner 部分代码使用中括号来分割变量类型与变量名，所以除了标示变量类型这一目的之外，变量名字中 **不要含有** ``]`` **符号！！**


.. _header-n74:

5. 定制化优化器
===============

1. 更换优化器

2. 多个优化器 or hook

3. grad norm/clip
