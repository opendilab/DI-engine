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
   - Atrai环境在该接口定义下的封装示例(pong, pendulum, cartpole)
   - 基于SUMO的交通信号灯控制环境在该接口定义下的封装示例
   - alphastar SC2环境再改接口定义下的封装示例

 3. model: 强化学习神经网络接口

 4. rl_utils: 强化学习算法库

   - td-lambda
   - vtrace(IMPALA)
   - UPGO
   - ppo
   - naive policy gradient
   - double dueling DQN
   - (TODO) SAC
   - (TODO)A2C
   - (TODO)MCTS

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
   - 计算图(computation_graph)
   - 数据生成器(actor)
   - 模型运行时容器(agent)
   - 向量化环境(env_manager)

 9. system: 系统控制模块

   - 运行信息管理(coordinator)
   - 跨集群通信(manager)

 10. entry: 启动入口模块

 11. docs: 文档

 12. tests: 单元测试相关


算法训练入口示例
--------------------

    完成安装之后，进入 ``nervex/entry`` 目录，找到 ``sumo_dqn_main.py`` 文件,
    即为在SUMO环境上运行的DQN算法示例（需要安装SUMO环境，配置SUMO_HOME环境变量，后续还会给出基于Atari环境的入口示例)。
    
    想要进行一组实验时，参照同目录下的 ``sumo_queue_len`` 文件夹，创建单独的实验文件夹，复制相应的执行脚本 ``run.sh`` 和配置文件 ``xxx.yaml`` 到实验文件夹下，修改配置文件中的参数，满足实验要求（例如在集群上运行时设置 ``use_cuda: True`` ）。然后启动执行脚本即可。下面所示为在slurm集群上的启动脚本，其中 `$1` 是相应的集群分区名。

    .. code:: bash

        work_path=$(dirname $0)
        srun -p $1 --gres=gpu:1 python3 -u ../sumo_dqn_main.py\
            --config_path $work_path/sumo_dqn_default_config.yaml 
