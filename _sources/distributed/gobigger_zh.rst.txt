异步并行在 GoBigger 中的应用
===============================

.. image::
   images/gobigger.gif
   :align: center

`GoBigger <https://github.com/opendilab/GoBigger>`_ 是由 OpenDILab 开源的一款多智能体对抗竞技游戏环境。\
玩家（AI）控制地图中的一个或多个圆形球，通过吃食物球和其他比玩家球小的单位来尽可能获得更多重量，并需避免被更大的球吃掉。

`在比赛中 <https://www.datafountain.cn/competitions/549/>`_ 玩家可以通过提交规则或模型来参与多智能体对抗，争取最佳成绩。\
当训练 RL 模型时，由于整场 GoBigger 游戏需要消耗较长时间，所以往往评估阶段占据了很长的时间，而真正用于训练的时间只占据了很小的一部分。\
即使将评估调整为训练 1000 次以后再进行，仍然是一个不小的开销，所以我们需要利用异步并行来增加资源利用率。

将串行改为并行后的代码执行过程如下图所示：

.. image::
   images/gobigger_async_parallel.png
   :align: center

.. image::
   images/gobigger_tf.png
   :align: center

只要机器还有资源剩余，在改为异步并行后评估阶段对训练的影响几乎可以忽略不计，采集和训练的步骤可以保持连续，而不用在中间等待评估。\
从 tensorboard 的记录上也可以看到，异步并行的版本相对于串行版本，在样本采集和训练效率上都有很大的提升。
