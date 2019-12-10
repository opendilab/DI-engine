FAQ
=====================

.. toctree::
   :maxdepth: 2

**模型训练过程中突然爆炸**

试一下grad clipper。目前自动模式未充分验证，建议手动设置clip参数。

**模型刚开始loss很大**

改一下初始化，使用较小的std

**只想test没有训练集怎么跑？**

dataset不提供train字段即可

**报错MemoryError？**

爆内存，将workers改小

**关于warmup的策略？**

当warmup_epochs>0时，将会以tatal_batch_size=batch_size*world_size为系数调整学习率；当warmup_epochs==0时，不对学习率进行调整。支持浮点数个epoch

**V100训练会卡住？**

V100训练使用以后缀为"_det"的pytorch环境。

**训练过程中修改代码？**

dataloader采用spawn的方式fork子进程。实验主进程每次enumerate(dataloader)会fork子进程读取数据，子进程会重新load代码，如果load成功(没有语法错误)，则执行dataset部分的代码。因此若程序运行过程中修改dataset部分的代码，则子进程会执行新的dataset代码。我们建议：1. 不要修改dataset部分代码，如果必须修改，可以增量添加新的dataset模块而不是在现有代码上修改；2. 程序epoch切换的时刻确保所有代码没有语法错误，保证子进程load成功。

**FP16 训练不收敛?**

检查模型中是否有诸如softmax，sigmoid之类对数值精度敏感的操作，将其替换为fp32位操作（参考BN）。

