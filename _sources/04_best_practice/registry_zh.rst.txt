注册（registry）
=========================

在 DI-engine 中, 为了方便地通过配置文件启动训练任务, 我们 **建议** 您应该使用注册 ``Registry`` 机制去注册那些自己实现的模块。

目前，注册机制支持这些模块：

   - policy
   - env
   - model
   - reward_model
   - learner
   - buffer
   - serial_collector
   - parallel_collector
   - comm_learner
   - comm_collector
   - commander
   - league
   - player

下面我们将通过 ``Policy`` 来举例说明，在实现新策略时如何使用 ``Registry``。

1.  为新策略添加 ``Registry`` 装饰器 (decorator)。

   .. code:: python
      
      from ding.utils import POLICY_REGISTRY

      @POLICY_REGISTRY.register('dqn')
      class DQNPolicy(Policy):
          pass

2.  在配置文件中，列出新策略的名称和文件路径。

   在键值 ``type`` 中，写下策略的名称。

   在键值 ``import_names`` 中，写入文件路径。 ``import_names`` 被要求为 ``list``，并且它的每个元素都是一个由python导入的抽象路径（即我们可以在Python Idle中运行``import name1.name2``），例如：

      - ``ding.policy.dqn``
      - ``dizoo.atari.envs.atari_env``

   配置文件示例如下：
   
   .. code:: python

        create_config = dict(
            policy=dict(
                type='multi_head_dqn',
                import_names=['dizoo.common.policy.multi_head_dqn'],
            # ...
            )
        )



3.通过系统函数创建模块

   如果你想通过 DI-engine 的 ``serial_pipeline`` 启动训练任务， 例如， 使用 CLI ``ding -m XXX -c XXXX_config.py -s XX``, 或者调用 ``serial_pipeline``， 那么第 3 步可以忽略，因为这一步已经在 ``serial_pipeline`` 函数中完成.
   但是，如果你想编写自己的入口文件，你可以调用 create_policy 函数来创建你的策略。

   .. code:: python
      
      from ding.policy import create_policy

      cfg: dict
      dqn_policy = create_policy(cfg.policy)

此外，您可以使用 CLI ``ding -q <registry name>`` 来查找已在 DI-engine 核心代码中注册的模块。例如：

.. image:: images/ding_cli_registry_query.png
