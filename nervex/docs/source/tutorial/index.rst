Tutorial
===============================

.. toctree::
   :maxdepth: 2

代码结构概述
~~~~~~~~~~~~~~~

 1. data: 数据加载

   - 对于离线数据：使用类似PyTorch的 `dataset` + `dataloader` + `sampler` + `collate_fn` 模式
   - 对于在线数据：提供强化版的Priortized Replay Buffer，支持更多的数据评估和采样机制

 2. envs: 强化学习环境接口

   - 通用环境类接口
   - 通用环境静态和动态元素接口
   - 通用环境处理特征工程函数
   - (TODO) Atari和Mujoco环境在该接口定义下的封装示例

 3. model: 强化学习神经网络接口

 4. rl_utils: 强化学习算法库

   - td-lambda
   - vtrace(IMPALA)
   - UPGO
   - ppo
   - naive policy gradient
   - (TODO) double dueling DQN
   - (TODO) SAC

 5. torch_utils: PyTorch相关工具库

   - 神经网络库
   - 损失函数库
   - PyTorch数据转换库
   - 训练现场保存(checkpoint)

 6. utils: 通用模块库

   - 计时函数
   - 数据压缩
   - 多卡训练（封装linklink）
   - 文件系统（封装ceph）
   - 日志和可视化

 7. league: 自对弈训练算法模块(self-play)

   - league-player模型
   - PFSP(prioritized fictitious self-play)
   - uniform self-play

 8. worker: 系统运行模块

   - 训练学习器(learner)
   - 优化器(optimizer)
   - 数据生成器(actor)
   - 模型运行时容器(agent)
   - 向量化环境(env_manager)

 9. system: 系统控制模块

   - 运行信息管理(coordinator)
   - 跨集群通信(manager)

 10. entry(train): 启动入口模块

 11. docs: 文档

 12. tests: 单元测试相关
