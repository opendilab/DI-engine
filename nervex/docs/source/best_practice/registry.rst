Registry
============

在 nerveX 中，为了可以方便地使用 config 文件启动训练任务，我们 **建议** 对于自己实现的一些模块，利用 ``Registry`` 机制进行注册。

目前支持的模块包括：
   - policy
   - env
   - learner
   - serial_collector
   - parallel_collector
   - comm_learner
   - comm_collector
   - commander
   - league
   - player

下面以 ``Policy`` 为例，讲解自定义Policy时， ``Registry`` 的使用方法。

   1.  自定义 ``Policy`` 类，然后添加注册器

   .. code:: python
      
      from nervex.utils import POLICY_REGISTRY

      @POLICY_REGISTRY.register('dqn')
      class DQNPolicy(Policy):
         pass

   2.  在 config 里指明所需要创建的 ``Policy`` 的名字及文件路径

   在 ``type`` 字段，指明名字。

   在 ``import_names`` 字段，指明文件路径。我们要求 ``import_names`` 需为一个 ``list`` ，其中每个元素是一个python的绝对import路径，
   即可以在 Python Idle 内执行 ``import name1.name2`` ，例如：

      - ``nervex.policy.dqn``
      - ``app_zoo.atari.envs.atari_env``

   示例如下：
   
   .. code:: python

      policy=dict(
         type='dqn',
         import_names=['app_zoo.sumo.policy.sumo_dqn'],
         # ...
      )

   若用户仔细阅读源码，会发现若使用在 nerveX 核心代码（指 ``nervex/`` 路径下）中实现的 ``Policy`` （例如DQN PPO等），
   在 config 中是没有指明 ``import_names`` 的。但若是用户自行实现的 ``Policy``，则 **必须指明** ``import_names``。


   3. 使用时通过系统函数创建

   普通用户做完前两步就可以直接使用 ``nervex -m XXX -c XXXX_config.py -s XX`` 启动任务了。因为“使用系统函数创建”这一步已经集成在了
   ``serial_pipeline`` 中。但如果用户有自定义 pipeline 的需求，可以通过 ``create_policy`` 函数创建自定义的Policy：

   .. code:: python
      
      from nervex.policy import create_policy

      cfg: dict
      dqn_policy = create_policy(cfg.policy)

此外，可以通过 ``nervex -q <registry name>`` 来查看在 nerveX 核心代码中已经注册的模块，例如：

.. image:: ./nervex_cli_query_registry.png


