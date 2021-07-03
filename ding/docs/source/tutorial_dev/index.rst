===============================
Tutorial-Developer
===============================

.. toctree::
   :maxdepth: 2

代码结构概述
===============

nervex (框架核心)
-----------------

    .. code:: bash

        nervex
        ├── config (配置文件及其工具)
        │   ├── buffer_manager.py (buffer manager配置文件)
        │   ├── config.py (Config类)
        │   ├── league.py (league配置文件)
        │   ├── parallel.py (并行配置文件)
        │   ├── serial.py (串行配置文件)
        │   └── utils.py (配置文件工具)
        ├── data (数据加载)
        │   ├── buffer_manager.py (多buffer管理)
        │   ├── collate_fn.py (数据处理函数)
        │   ├── dataloader.py (异步数据加载器)
        │   └── structure (所需数据结构)
        ├── design (设计图)
        ├── docs (文档)
        ├── entry (启动入口)
        │   ├── cli.py (命令行)
        │   ├── parallel_entry.py (并行入口)
        │   └── serial_entry.py (串行入口)
        ├── envs (强化学习环境接口)
        │   ├── common (通用环境元素基类)
        │   ├── env (环境基类和具体的环境类)
        │   └── env_manager (环境管理器类)
        ├── hpc_rl (加速算子组件)
        │   ├── hpc_rll-0.0.1-cp36-cp36m-linux_x86_64.whl (环境库打包whl文件)
        │   └── wrapper.py
        ├── interaction (独立于业务的交互式服务框架)
        │   ├── base
        │   ├── config
        │   ├── exception
        │   ├── master
        │   └── slave
        ├── league (联盟训练决策调度模块)
        │   ├── algorithm.py
        │   ├── base_league.py
        │   ├── league_wrapper.py
        │   ├── payoff.py
        │   ├── player.py
        │   ├── shared_payoff.py
        │   ├── solo_league.py
        │   └── starcraft_player.py
        ├── loader (数据组合框架组件)
        │   ├── base.py
        │   ├── collection.py
        │   ├── dict.py
        │   ├── exception.py
        │   ├── mapping.py
        │   ├── norm.py
        │   ├── number.py
        │   ├── string.py
        │   ├── tests
        │   ├── types.py
        │   └── utils.py
        ├── model (强化学习神经网络接口)
        │   ├── actor_critic
        │   ├── atoc
        │   ├── coma
        │   ├── common
        │   ├── common_arch
        │   ├── discrete_net
        │   ├── qac
        │   ├── qmix
        │   ├── sac
        │   └── model_wrappers
        ├── policy (强化学习策略库)
        │   ├── a2c.py
        │   ├── base_policy.py
        │   ├── collaQ.py
        │   ├── coma.py
        │   ├── common_policy.py
        │   ├── ddpg.py
        │   ├── dqn.py
        │   ├── dqn_vanilla.py
        │   ├── impala.py
        │   ├── ppo.py
        │   ├── ppo_vanilla.py
        │   ├── qmix.py
        │   ├── r2d2.py
        │   ├── rainbow_dqn.py
        │   └── sac.py
        ├── rl_utils (强化学习工具库)
        │   ├── a2c.py
        │   ├── adder.py
        │   ├── beta_function.py
        │   ├── coma.py
        │   ├── exploration.py
        │   ├── gae.py
        │   ├── isw.py
        │   ├── ppo.py
        │   ├── td.py
        │   ├── tests
        │   ├── upgo.py
        │   ├── value_rescale.py
        │   └── vtrace.py
        ├── scripts (命令行脚本)
        │   ├── local_parallel.sh
        │   ├── local_serial.sh
        │   └── slurm_parallel.sh
        ├── torch_utils (PyTorch相关工具库)
        │   ├── checkpoint_helper.py (训练现场保存和加载)
        │   ├── data_helper.py (Tensor数据转换库)
        │   ├── distribution.py (概率分布库)
        │   ├── loss (损失函数库)
        │   ├── metric.py (距离度量库)
        │   ├── network (神经网络库)
        │   ├── nn_test_helper.py (神经网络测试库)
        │   └── optimizer_helper.py (优化器和梯度操作库)
        ├── utils
        │   ├── autolog (变量追踪工具)
        │   ├── collection_helper.py
        │   ├── compression_helper.py (数据压缩)
        │   ├── config_helper.py (配置文件读取与合并)
        │   ├── default_helper.py (数据变换函数)
        │   ├── design_helper.py (代码设计工具)
        │   ├── dist_helper.py (多卡训练)
        │   ├── fake_linklink.py (伪linklink)
        │   ├── file_helper.py （文件系统）
        │   ├── import_helper.py (库导入)
        │   ├── lock_helper.py (同步和互斥锁)
        │   ├── log_helper.py (日志和可视化)
        │   ├── slurm_helper.py (slurm工具)
        │   ├── system_helper.py (系统工具)
        │   └── time_helper.py （计时函数）
        └── worker
            ├── collector (数据生成器)
            ├── adapter (适配器)
            ├── coordinator (协作器)
            └── learner (训练学习器)

app_zoo (基于nerveX的DRL应用)
-----------------------------

    .. code:: bash

        app_zoo
        ├── atari
        ├── classic_control
        │   ├── bitflip
        │   ├── cartpole
        │   └── pendulum
        ├── gfootball (multi-agent football)
        ├── mujoco
        ├── multiagent_particle
        ├── smac
        └── sumo (traffic light control)


数据流图
============================

nerveX每一个训练实例可以主要分为三部分，即Coordinator(协作器)，Learner(学习器)，Collector(数据生成器)，再加上用于数据通信的Middleware(中介组件)，这四个模块的详细组成及之间的数据关系如下图所示:

.. image:: dataflow.png


而上述几个核心模块之间的数据流序列关系可以参考下图:

.. image:: flow_sequence.png
