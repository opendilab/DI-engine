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
~~~~~~~~~~~~~~~~~~

    完成安装之后，进入 ``nervex/entry`` 目录，找到 ``sumo_dqn_main.py`` 文件,
    即为在SUMO环境上运行的DQN算法示例（需要安装SUMO环境，配置SUMO_HOME环境变量，后续还会给出基于Atari环境的入口示例)。
    
    想要进行一组实验时，参照同目录下的 ``sumo_queue_len`` 文件夹，创建单独的实验文件夹，复制相应的执行脚本 ``run.sh`` 和配置文件 ``xxx.yaml`` 到实验文件夹下，修改配置文件中的参数，满足实验要求（例如在集群上运行时设置 ``use_cuda: True`` ）。然后启动执行脚本即可。下面所示为在slurm集群上的启动脚本，其中 `$1` 是相应的集群分区名。

    .. code:: bash

        work_path=$(dirname $0)
        srun -p $1 --gres=gpu:1 python3 -u ../sumo_dqn_main.py\
            --config_path $work_path/sumo_dqn_default_config.yaml 


DRL快速上手指南
~~~~~~~~~~~~~~~~
深度强化学习(DRL)在很多问题场景中展现出了媲美甚至超越人类的性能，本指南将从DRL的启明星——DQN开始，逐步介绍如何使用nerveX框架在Cartpole游戏环境上训练一个DQN智能体，主要将分为如下几个部分：

 - 创建环境
 - 搭建神经网络
 - 搭建强化学习训练策略
 - 搭建数据交互生成器
 - 其他功能拓展

创建环境
^^^^^^^^^^^^
RL不同于传统的监督学习，数据是一般离线准备好的，它需要实时让智能体和问题环境进行交互，产生数据帧用于训练。nerveX为了处理实际问题场景中复杂的环境结构定义，抽象了环境及其基本元素相关模块（`Env Overview
<../package_ref/env/env_overview.html>`_），该抽象定义了环境和外界交互的相关接口，数据帧中每个元素的格式和取值范围等基本信息。对于Cartpole环境，nerveX已经完成实现，可以通过如下的代码直接调用：

.. code:: python

    from nervex.envs.gym import CartpoleEnv

    env = CartpoleEnv(cfg={})  # use default env config

为了加快生成数据的效率，nerveX提供了向量化环境运行的机制，并由 ``Env Manager`` （环境管理器） 模块负责维护相关功能，每次运行批量启动多个环境交互生成数据。注意到这里会存在对于环境输出和输入数据进行打包和拆包的操作（数据相关），故需要 **环境内部**
提供相应的打包拆包函数，供环境管理器调用。除此之外，环境管理器与环境本身完全解耦，无需了解任何环境具体的数据信息。系统提供了多种实现方式的环境管理器，最常用的子进程环境管理器的实例代码如下：

.. code:: python
    
    from nervex.worker.actor.env_manager import SubprocessEnvManager

    # create 4 Cartpole env with default config(set `env_cfg=[{} for _ in range(4)]`)
    env_manager = SubprocessEnvManager(env_fn=CartpoleEnv, env_cfg=[{} for _ in range(4)], env_num=4)


.. note::

    向量化环境目前支持用不同的配置创建同类环境（例如地图不同的多个SC2环境），不支持向量化环境中存在不同类模型，如果有此类需求请创建多个环境管理器

搭建神经网络
^^^^^^^^^^^^^^

nerveX基于PyTorch深度学习框架搭建所有的神经网络相关模块，支持用户自定义各式各样的神经网络，不过，nerveX也根据RL等决策算法的需要，构建了一些抽象层次和API，主要分为 ``model`` （模型）和 ``agent`` （智能体）两部分。

模型部分是对一些经典算法的抽象，比如对于Actor-Critic系列算法和Dueling DQN算法，nerveX为其实现了相关的模型基类，其他部分均可由用户根据自己的需要自定义实现。对于在Cartpole上最简单版本的DQN，示例代码如下：

.. code:: python

    import torch
    import torch.nn as nn
    

    # network definition
    class FCDQN(nn.Module):
        def __init__(self, input_dim, action_dim, hidden_dim_list=[128, 256, 256], device='cpu'):
            super(FCDQN, self).__init__()
            self.act = nn.ReLU()
            layers = []
            for dim in hidden_dim_list:
                layers.append(nn.Linear(input_dim, dim))
                layers.append(self.act)
                input_dim = dim
            self.main = nn.Sequential(*layers)
            self.action_dim = action_dim
            self.pred = nn.Linear(input_dim, action_dim)
            self.device = device

        def forward(self, x, info={}):
            x = self.main(x)
            x = self.pred(x)
            return x

    # create network
    env_info = env.info()
    obs_shape = env_info.obs_shape.shape
    act_shape = env_info.act_shape.shape
    model = FCDQN(obs_shape, act_shape)

.. note::

    注意由于Cartpole是一个离散动作空间环境，神经网络的输出并不是具体的动作值，而是对于整个动作空间选取动作的logits，其将会在其他模块中完成采样操作转化成具体的动作

智能体部分是对模型运行时行为的抽象（例如根据eps-greedy方法对logits进行采样，对于使用RNN的神经网络维护其隐状态等），具体的设计可以参考 `Agent Overview <../package_ref/worker/agent/agent_overview.html>`_ 。由于模型可能在多个系统组件内通过不同的方式使用，nerveX使用 ``Agent Plugin`` （智能体插件）的定义不同的功能，并为各个组件内的模型添加相应的插件，完成定制化。对于Cartpole DQN，相应的智能体示例代码如下, 其中Learner和Actor分别代码训练端和数据生成端：

.. code:: python

    from nervex.worker.agent import BaseAgent, IAgentStatelessPlugin
    from collections import OrderedDict
    

    class CartpoleDqnLearnerAgent(BaseAgent):
        def __init__(self, model: torch.nn.Module, plugin_cfg: dict) -> None:
            self.plugin_cfg = OrderedDict({
                'grad': {
                    'enable_grad': True
                },
            })
            # whether use double(target) q-network plugin
            if plugin_cfg['is_double']:
                self.plugin_cfg['target_network'] = {'update_cfg': {'type': 'momentum', 'kwargs': {'theta': 0.99}}}
            super(CartpoleDqnLearnerAgent, self).__init__(model, self.plugin_cfg)


    class CartpoleDqnActorAgent(BaseAgent):
        def __init__(self, model: torch.nn.Module) -> None:
            plugin_cfg = OrderedDict(
                {
                    'eps_greedy_sample': {},
                    'grad': {
                        'enable_grad': False
                    },
                }
            )
            super(CartpoleDqnActorAgent, self).__init__(model, plugin_cfg)

搭建强化学习训练策略
^^^^^^^^^^^^^^^^^^^^^

搭建数据交互生成器
^^^^^^^^^^^^^^^^^^
