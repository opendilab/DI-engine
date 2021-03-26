===============================
Tutorial
===============================

.. toctree::
   :maxdepth: 2

代码结构概述
===============

nervex (框架核心)
-----------------

    .. code:: bash

        nervex
        ├── armor (模型运行时容器)
        │   ├── armor.py (BaseArmor及Armor类)
        │   └── armor_plugin.py (armor插件)
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
        │   └── env (环境基类和具体的环境类)
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
        │   └── sac
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
            ├── actor (数据生成器)
            ├── adapter (适配器)
            ├── coordinator (协作器)
            └── learner (训练学习器)

app_zoo (基于nerveX的DRL应用)
-----------------------------

    .. code:: bash

        app_zoo
        ├── alphastar (SC2)
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

nerveX每一个训练实例可以主要分为三部分，即Coordinator(协作器)，Learner(学习器)，Actor(数据生成器)，再加上用于数据通信的Middleware(中介组件)，这四个模块的详细组成及之间的数据关系如下图所示:

.. image:: dataflow.png


而上述几个核心模块之间的数据流序列关系可以参考下图:

.. image:: flow_sequence.png


算法训练入口示例(串行版本)
=================================

训练脚本及其启动
------------------------------
    
    完成安装之后，可以仿照 ``nervex/entry/tests/test_serial_entry.py`` 文件中单元测试的写法，创建一个训练脚本并命名为 ``cartpole_dqn.py``：

    .. code:: python

        from copy import deepcopy
        from nervex.entry import serial_pipeline
        from app_zoo.classic_control.cartpole.entry import cartpole_dqn_default_config
        config = deepcopy(cartpole_dqn_default_config)
        serial_pipeline(config, seed=0)

    如以上代码，就是读取了 ``app_zoo`` 中的 ``cartpole_dqn_default_config.py`` 配置文件，并传入 ``serial_pipeline`` 开始训练。

    .. note::

        入口函数 ``serial_pipeline`` 还支持指定自定义的 **环境**, **策略**, **神经网络模型**, 具体使用方式可以参见QA部分或是直接查看相关代码。
    
    想要进行一组实验时，应 **创建单独的实验文件夹，复制相应的执行脚本（如有必要，也需一同复制配置文件）到实验文件夹下** ，然后启动执行脚本即可。

    下面所示为一般本地测试时的启动脚本

    .. code:: bash

        python3 -u cartpole_dqn.py

    下面所示为在slurm集群上的启动脚本，其中 `$1` 是相应的集群分区名。

    .. code:: bash

        srun -p $1 --gres=gpu:1 python3 -u cartpole_dqn.py

    .. note::

        如果是使用 ``pip install .`` 命令安装，即未指定-e，还可以通过命令行调用串行训练入口：

        .. code:: bash

            nervex -m serial -c config.yaml -s 0
            nervex -m serial -c config.py -s 0
        
        此处 config 文件支持 yaml 或是 py 格式，但若为 py 格式，需要声明 ``main_config`` 变量，具体说明请见下一节 **配置文件** 。

配置文件
--------

    根据不同的需求，可以修改配置文件并自定义相关的启动脚本，配置文件中可能修改的地方主要有如下几处：

      - policy.use_cuda: 是否使用cuda，主要取决于使用者的机器上是否有GPU
      - env.env_type: 如要更改所使用的环境，首先修改env.env_type，并对应修改env.import_names，atari及mujuco还需修改env.env_id，不同环境的evaluator.stop_val可能不同也需要修改。需注意环境的observation是图像还是向量，并检查是否需要对应修改policy.model中的encoder。
      - policy: 若要更改所使用的算法/策略，首先修改policy.policy_type，并对应修改policy.import_names, policy.on_policy, policy.model等。

    .. note::

        无论是串行还是并行版本的 config 文件，若是 py 格式，且希望通过命令行的方式启动脚本，请务必在文件中声明 ``main_config`` 变量，
        令其等于真实的 ``EasyDict`` 类型的配置变量，如下：

        .. code:: python

            cartpole_dqn_default_config = dict(
                # ...
            )
            cartpole_dqn_default_config = EasyDict(cartpole_dqn_default_config)
            main_config = cartpole_dqn_default_config

运行后产生的文件
---------------------

    串行版本运行起来后会在当前目录产生 ``ckptBaseLearner*`` 及 ``log`` 两个文件夹，分别存放 checkpoint 及 log 文件，文件树如下：

    .. code:: bash
        
        ./
        ├── cartpole_a2c_default_config.py
        ├── ckptBaseLearner140403751719992
        │   ├── iteration_0.pth.tar
        │   └── iteration_200.pth.tar
        └── log
            ├── actor
            │   └── collect_logger.txt
            ├── buffer
            │   └── armor_buffer
            │       ├── armor_logger.txt
            │       └── armor_tb_logger
            ├── evaluator
            │   ├── evaluator_logger.txt
            │   └── evaluator_tb_logger
            └── learner
                ├── learner_logger.txt
                └── learner_tb_logger

    对于 ``ckptBaseLearner*`` ，一般来说，iteration 最大的文件保存有 evaluate 阶段 reward 最高的模型， iteration 从小至大的 eval_reward 也应当是从小至大的。

    ``log`` 下包括 ``actor``, ``evaluator``, ``learner``, ``buffer`` 四个文件夹，除了 ``actor`` 外，均既有 tensorboard logger 又有 text logger，
    而 ``actor`` 仅有 text logger。这些 logger 均按照各自的 log_freq 在一定的时间/步数间隔下进行记录。

    ``actor`` 记录与环境交互的信息， ``learner`` 记录根据数据进行策略更新的信息， ``evaluator`` 记录对于当前最新策略的评估信息，
    ``buffer`` 记录数据被塞入与采样出的各种统计量。

算法训练入口示例(并行版本)
=================================

训练脚本及其启动
------------------

    进入 ``app_zoo/classic_control/cartpole/entry/parallel`` 目录，找到 ``cartpole_dqn_default_config.py`` 文件,
    即为在Cartpole环境上运行的并行训练配置文件。

    下面所示为一般本地测试时的启动脚本

    .. code:: bash

        nervex -m parallel -c cartpole_dqn_default_config.py -s 0
    
    下面所示为在slurm集群上的启动脚本，其中需要指定actor和learner相应的计算节点IP，Coordinator默认运行在管理节点上。
    
    .. code:: bash

        nervex -m parallel -p slurm -c cartpole_dqn_default_config.py -s 0 --actor_host SH-IDC1-10-198-8-66 --learner_host SH-IDC1-10-198-8-66
    
    nervex 命令参数选项:

        - **\-v, --version** : Show package's version information.
        - **\-m, --mode [serial|parallel|eval]** : serial or parallel or eval
        - **\-c, --config TEXT** : Path to DRL experiment config
        - **\-s, --seed INTEGER** : random generator seed(for all the possible package: random, numpy, torch and user env)
        - **\-p, --platform [local|slurm|k8s]** : local or slurm or k8s
        - **\-ch, --coordinator_host TEXT** : coordinator host
        - **\-lh, --learner_host TEXT** : learner host
        - **\-ah, --actor_host TEXT** : actor host
        - **\-h, --help** : Show this message and exit.

配置文件
--------
    
    根据不同的使用环境，可以相应修改配置文件，其中可能修改的地方主要有如下几处：

      - use_cuda: 是否使用cuda，主要取决于使用者的机器上是否有GPU，注意这时的启动脚本要指定cuda device相关
      - use_distributed: 是否使用多机多卡训练，主要取决于使用者是否安装了linklink，以及是否要开启多机多卡训练，注意这时的启动脚本中要指定 `mpi` 相关
      - repeat_num: learner端参与训练的GPU卡数，目前仅支持单机，最大值为一台机器上空闲的GPU数目
      - path_armor等: 这些字段是多机版本训练进行数据通信的相关路径，默认使用当前目录，即通过文件系统进行通信，在集群上一般使用ceph，需要进行相关配置并对应更改这些字段

运行后产生的文件
---------------------
    
    并行版本运行起来后会在当前目录产生 ``log`` 和 ``data`` 两个文件夹，以及 ``policy_*`` 文件，文件树如下：

    .. code:: bash

        ./
        ├── __init__.py
        ├── cartpole_dqn_default_config.py
        ├── data
        │   ├── env_0_1f03b27a-68f3-11eb-9a9b-29face2f0d06
        │   ├── env_1_2c996e0a-68f3-11eb-9a9b-29face2f0d06
        │   ├── ....
        │   └── env_7_4939d342-68f3-11eb-9a9b-29face2f0d06
        ├── log
        │   ├── actor
        │   │   ├── 011f43e3-6d93-4e6d-ab6a-f124b1719050_476275_logger.txt
        │   │   ├── 34bc401b-ae5b-4a0c-816c-1db81738ae8c_606251_logger.txt
        │   │   ├── ....
        │   │   └── d8b1ce8f-f6ce-4d20-8085-7f2d9ce5bea8_476962_logger.txt
        │   ├── buffer
        │   │   └── armor_buffer
        │   │       ├── armor_logger.txt
        │   │       └── armor_tb_logger
        │   ├── commander
        │   │   ├── commander_logger.txt
        │   │   └── commander_tb_logger
        │   ├── coordinator_logger.txt
        │   ├── evaluator
        │   │   ├── 099d882b-ac35-4e77-a85b-0ec4924ce45a_160479_logger.txt
        │   │   ├── 0c11e0e2-6b5b-417d-968c-ddc205a819c0_297009_logger.txt
        │   │   ├── ....
        │   │   └── fef38c77-1fa6-4d62-a0a3-5a904753e931_695838_logger.txt
        │   └── learner
        │       ├── learner_logger.txt
        │       └── learner_tb_logger
        └── policy_587ffbea-31bc-4aac-8d60-70ba68f5c5a7_611148

    ``policy_*`` 是由 learner 存储，由 actor 读入以更新策略用的。

    ``data`` 下存储的是 replay buffer 中的 trajectory（replay buffer仅存储这些 trajectory 的路径，而不实际存储数据）。

    ``log`` ，其下包括 ``actor``, ``evaluator``, ``learner``, ``buffer``, ``commander`` 五个文件夹，以及 ``coordinator_logger.txt`` 文件。
    其中， ``actor``, ``evaluator`` 会按照不同的 task 生成多个 txt 文件； ``learner`` 部分与串行版本类似，多个 task 的文字记录均在同一 txt 文件中，
    但 tensorboard 会分 task 记录。 ``buffer`` 与串行版本相同。 ``commander`` 中将 evaluator 中的信息进行了整合，方便用户查看当前策略训练情况。
    ``coordinator_logger.txt`` 则记录了和并行模式下通信相关的各种信息。


DRL快速上手指南(串行版本)
==============================
深度强化学习(DRL)在很多问题场景中展现出了媲美甚至超越人类的性能，本指南将从DRL的启明星——DQN开始，逐步介绍如何使用nerveX框架在Cartpole游戏环境上训练一个DQN智能体，主要将分为如下几个部分：

 - 环境相关
 - 神经网络模型相关
 - 优化目标(损失函数)相关
 - 数据队列相关
 - 策略(Policy)相关
 - 其他功能拓展

完整的入口文件可以参见 ``nervex/entry/serial_entry.py``

.. note::

    注意一个深度强化学习算法可能包括神经网络模型，运行计算图(训练/数据生成)，优化目标(损失函数)，优化器等多个部分，nerveX在实现上将各个模块进行了解耦设计，所以相关代码可能较为分散，但一般的代码组织体系为：model（神经网络模型），rl_utils（具体的强化学习优化目标函数），以及两种可选功能组件Armor（神经网络模型在训练/数据生成/测试时的不同动态行为，例如RNN隐状态的维护，Double DQN算法中target network的维护），Adder（将收集到的数据帧整合成训练所需的状态），以及将上述各个模块组织串联起来，完整的强化学习策略定义，Policy模块（例如DQNPolicy）。

环境相关
-----------

RL不同于传统的监督学习中可以离线准备数据，RL需要 **实时** 让智能体和问题环境进行交互，产生数据帧用于训练。
nerveX为了处理实际问题场景中复杂的环境结构定义，抽象了环境及其基本元素相关模块（`Env Overview <../feature/env_overview.html>`_）。
该抽象定义了环境和外界交互的相关接口，数据帧中每个元素的格式和取值范围等基本信息。
对于CartPole环境，nerveX已经完成实现，可以通过如下的代码直接调用：

.. code:: python

    from app_zoo.classic_control.cartpole.envs import CartPoleEnv

    env = CartPoleEnv(cfg={})  # use default env config

而在 ``serial_pipeline`` 中，我们有两种创建环境的方式，第一种是通过 ``cfg.env`` ，即配置文件中 ``env`` 相关字段进行自动创建，第二种是通过 ``env_setting`` 参数直接从调用者处得到环境类，actor部分的环境配置，以及evaluator部分的环境配置，具体的代码如下：

.. code:: python

    if env_setting is None:
        env_fn, actor_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, actor_env_cfg, evaluator_env_cfg = env_setting
    em_type = cfg.env.env_manager_type
    if em_type == 'base':
        env_manager_type = BaseEnvManager
    elif em_type == 'aynsc_subprocess':
        env_manager_type = SubprocessEnvManager
    elif em_type == 'subprocess':
        env_manager_type = SyncSubprocessEnvManager

其中从config中获取env_setting的方式为 ``get_vec_env_setting`` 函数：

.. code:: python
    
    def get_vec_env_setting(cfg: dict) -> Tuple[type, List[dict], List[dict]]:
        import_module(cfg.pop('import_names', []))
        if cfg.env_type in env_mapping:
            env_fn = env_mapping[cfg.env_type]
        else:
            raise KeyError("invalid env type: {}".format(cfg.env_type))
        actor_env_cfg = env_fn.create_actor_env_cfg(cfg)
        evaluator_env_cfg = env_fn.create_evaluator_env_cfg(cfg)
        return env_fn, actor_env_cfg, evaluator_env_cfg

注意到我们对 ``actor_env_cfg`` , ``evaluator_env_cfg`` 进行了分开处理，这是考虑到训练过程中为了取得更好的训练效果，例如在Atari环境中经常会使用 ``Wrapper``
对环境做不同的处理，而 ``Wrapper`` 处理后的 ``evaluator_env`` 其实并不能很好的衡量算法的效果，所以需要区别对待。

为了加快生成数据的效率，nerveX提供了向量化环境运行的机制，即一次运行多个同类环境进行交互生成训练数据，并由 ``Env Manager`` （环境管理器） 模块负责维护相关功能，每次运行批量启动多个环境交互生成数据。环境管理器与环境本身内容完全解耦，无需了解任何环境具体的数据信息，环境本身可以使用任意数据类型，但经过环境管理器处理之后，进入nervex一律为PyTorch Tensor相关数据格式。系统提供了多种实现方式的环境管理器，最常用的子进程环境管理器的实例代码如下：

.. code:: python
    
    from nervex.worker.actor.env_manager import SubprocessEnvManager

    # create 4 CartPoleEnv env with default config(set `env_cfg=[{} for _ in range(4)]`)
    env_manager = SubprocessEnvManager(env_fn=CartPoleEnv, env_cfg=[{} for _ in range(4)], env_num=4)

我们在 ``serial_pipeline`` 中，通过 ``config`` 文件中对应的 ``cfg.env.env_manager_type`` 控制使用 ``SyncSubprocessEnvManager``, ``SubprocessEnvManager``
还是 ``BaseEnvManager`` 。

.. note::

    向量化环境目前支持用不同的配置创建同类环境（例如地图不同的多个SC2环境），不支持向量化环境中存在不同类模型，如果有此类需求请创建多个环境管理器

神经网络模型相关
--------------------

nerveX基于PyTorch深度学习框架搭建所有的神经网络相关模块，支持用户自定义各式各样的神经网络，不过，nerveX也根据RL等决策算法的需要，构建了一些抽象层次和API，主要分为 ``Model`` （模型）和 ``Armor`` （运行时模型）两部分，若已有的Armor组件无法满足需求，使用者也可以完全自定义相关的代码段，其和训练主体代码并无耦合。

模型部分是对一些经典算法的抽象，比如对于Actor-Critic系列算法和Dueling DQN算法，nerveX为其实现了相关的模型基类，并且进行了多层的模块化的封装，详见 
``nervex/model/discrete_net/discrete_net.py`` 和其对应的测试文件 ``nervex/model/discrete_net/test_discrete_net.py`` 。

用户也可根据自己的需要自定义实现，示例代码如下：

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
            x = x['obs']
            x = self.main(x)
            x = self.pred(x)
            return {'logit': x}

    # create network
    env_info = env.info()
    obs_shape = env_info.obs_space.shape
    act_shape = env_info.act_space.shape
    model = FCDQN(obs_shape, act_shape)

.. note::

    此处实现的 ``FCDQN`` 示例网络其实相当于 ``nervex/model/discrete_net/discrete_net.py`` 中的 ``FCDiscreteNet``。

.. note::

    注意由于Atari是一个离散动作空间环境，神经网络的输出并不是具体的动作值，而是对于整个动作空间选取动作的logits，其将会在其他模块中完成采样操作转化成具体的动作。

.. note::

    nerveX的model模块中实现更为复杂的DQN（支持不同 ``Encoder`` 和使用 ``LSTM`` ），使用者可使用自定义所用的神经网络，或内置版本的神经网络。
    内置版本的神经网络中，以 ``FC`` 开头表示使用接受 ``1-dim`` 的obs输入 ``Encoder`` ，以 ``Conv`` 开头表示使用接受 ``[Channel, Hight, Width]`` 的输入的 ``Encoder`` ，
    包含 ``R`` 的表示带有含 ``LSTM`` 的Recurrent Network。


.. tip::

    为了便于和其他模块的对接，nerveX限制神经网络的输入输出为dict形式，即键为字符串值为Tensor或一组Tensor。但dict确实存在无法明晰输入输出数据具体内容的问题，故建议使用者为自己的神经网络准备
    相应的单元测试，并在forward方法中注明输入和输出的数据键及值的Tensor维度，格式可参考 `https://gitlab.bj.sensetime.com/open-XLab/cell/nerveX/blob/master/nervex/rl_utils/ppo.py#L32`。

Armor 部分是对模型运行时行为的抽象（例如根据eps-greedy方法对logits进行采样，对于使用RNN的神经网络维护其隐状态等），具体的设计可以参考 `Armor Overview <../feature/armor_overview.html>`_ 。由于一个神经网络模型可能在多个系统组件内通过不同的方式使用（训练/数据生成/测试），nerveX使用 ``Armor Plugin`` （插件）的定义不同的功能，并为各个组件内的模型添加相应的插件，完成定制化。对于CartPole DQN，使用系统预设的默认DQN Armor即可，示例如下， 其中Learner和Actor分别代码训练端和数据生成端：


.. note::

   如果使用者想要定义自己的armor，请参考 `Armor Overview <../feature/armor_overview.html>`_ 中相关内容。如果使用者觉得Armor的现有设计和实现无法满足需求，也可以自定义完成相应的功能，nerveX并不强制要求使用Armor。

优化目标(损失函数)相关
-------------------------
在nerveX中，构建算法训练需要执行相应前向计算过程得到优化目标（loss）的模块，负责将预处理好后的数据合理地送入模型进行处理，之后使用模型输出结果计算该次迭代的优化目标，返回相关结果。

优化目标(损失函数)相关的内容在 ``nervex/rl_utils`` 中可以找到，如DQN算法就需要使用基于Q值的td error， 如 ``nervex/rl_utils/td.py`` 中的函数：

.. code:: python

    q_nstep_td_data = namedtuple(
        'q_nstep_td_data', ['q', 'next_n_q', 'action', 'next_n_action', 'reward', 'done', 'weight']
    )

.. code:: python

    def q_nstep_td_error(
            data: namedtuple,
            gamma: float,
            nstep: int = 1,
            criterion: torch.nn.modules = nn.MSELoss(reduction='none'),
    ) -> torch.Tensor:
        r"""
        Overview:
            Multistep (1 step or n step) td_error for q-learning based algorithm
        Arguments:
            - data (:obj:`q_nstep_td_data`): the input data, q_nstep_td_data to calculate loss
            - gamma (:obj:`float`): discount factor
            - criterion (:obj:`torch.nn.modules`): loss function criterion
            - nstep (:obj:`int`): nstep num, default set to 1
        Returns:
            - loss (:obj:`torch.Tensor`): nstep td error, 0-dim tensor
        Shapes:
            - data (:obj:`q_nstep_td_data`): the q_nstep_td_data containing\
                ['q', 'next_n_q', 'action', 'reward', 'done']
            - q (:obj:`torch.FloatTensor`): :math:`(B, N)` i.e. [batch_size, action_dim]
            - next_n_q (:obj:`torch.FloatTensor`): :math:`(B, N)`
            - action (:obj:`torch.LongTensor`): :math:`(B, )`
            - next_n_action (:obj:`torch.LongTensor`): :math:`(B, )`
            - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is timestep(nstep)
            - done (:obj:`torch.BoolTensor`) :math:`(B, )`, whether done in last timestep
        """
        q, next_n_q, action, next_n_action, reward, done, weight = data
        assert len(action.shape) == 1, action.shape
        if weight is None:
            weight = torch.ones_like(action)

        batch_range = torch.arange(action.shape[0])
        q_s_a = q[batch_range, action]
        target_q_s_a = next_n_q[batch_range, next_n_action]

        target_q_s_a = nstep_return(nstep_return_data(reward, target_q_s_a, done), gamma, nstep)
        return (criterion(q_s_a, target_q_s_a.detach()) * weight).mean()

所有 ``loss`` 相关的算法都是使用某种 ``namedtuple`` 作为计算的输入格式。

搭建数据队列
-------------
学习器和数据生成器通过数据队列进行数据帧的交互，该模块除了简单的先入先出队列之外，还集成了一些数据质量分析和数据采样的相关操作，具体使用的样例如下：

.. code:: python

    from nervex.data import BufferManager


    # you can refer to `nervex/data/replay_buffer_default_config.yaml` for the detailed configuration 
    cfg = {'meta_maxlen': 10}
    buffer_ = BufferManager(cfg)

    # add 10 data
    for _ in range(10):
        buffer_.push_data({'data': 'placeholder'})
    data = buffer_.sample(4)  # sample 4 data

而在 ``serial_pipeline`` 中，我们通过 ``cfg.replay_buffer`` 对 ``replay_buffer`` 自动进行了创建：

.. code:: python

    replay_buffer = BufferManager(cfg.replay_buffer)

创建策略
--------
nerveX已经实现了诸多DRL常用算法，使用者可在配置文件中指定需要使用的RL算法名以及相应的模块名，创建policy的代码如下：

.. code:: python

    policy = create_policy(cfg.policy)

如果使用者想要自定义策略，可以参见文档QA中的说明进行实现，并指定 ``serial_pipeline`` 的 ``policy_fn`` 参数传入该自定义类

DRL Policy Example(DQN)
--------------------------------------------------

在撰写DQN Policy之前，我们先 ``import`` 需要的模块


.. code:: python

    #引入typing类规范格式
    from typing import List, Dict, Any, Tuple, Union, Optional

.. code:: python

    #我们的模型框架基于PyTorch
    import torch

.. code:: python

    #我们的环境返回的timestep基于nametuple，训练过程中的trajectory则是放在deque中
    from collections import namedtuple, deque
    
    #我们的transition data是EasyDict格式
    from easydict import EasyDict

.. code:: python

    #默认继承和扩展了torch.optim类的optimizer（集成各种梯度处理操作），也可以自由选用其他优化器
    from nervex.torch_utils import Adam

.. code:: python

    #DQN是q value相关的算法，因此引入q值相关的loss计算函数
    from nervex.rl_utils import q_1step_td_data, q_1step_td_error, q_nstep_td_data, q_nstep_td_error
    
    #epsilon_greedy for exploration
    from nervex.rl_utils import epsilon_greedy
    
    #Adder用于处理actor产生的数据，生成训练所需的数据内容（Adder是可选使用模块，使用者也可自定义相应的处理模块）
    from nervex.rl_utils import Adder

.. code:: python

    #Armor模块，神经网络的运行时容器，为神经网络在不同使用场景下提供相应功能，包括用于更新策略的learner部分和用于collect数据的actor部分以及用于eval的evaluator部分
    #(Armor是可选使用模块，使用者也可自定义相应的处理模块)
    #Armor具体的使用方式可以参照下面代码中的实例
    from nervex.armor import Armor

.. code:: python

    #算法使用的model，通常为神经网络
    from nervex.model import FCDiscreteNet, ConvDiscreteNet

.. code:: python

    #引入Policy基类
    from .base_policy import Policy, register_policy
    from .common_policy import CommonPolicy

下面以DQN Policy为例讲解如何构建一个新的Policy类 DQN
Policy中只需实现与具体算法策略相关的内容，其编写需要实现几个部分：

 - 算法使用的 ``model``，通常为神经网络， 即 ``self._model``
 - 算法进行神经网络优化的部分(learn)，通常包括优化器，即 ``self._optimizer`` ，训练优化前的数据处理，训练优化的整个计算图(forward)，包括前向传播计算得到损失函数，梯度反向传播进行参数更新，其他信息的更新
 - 算法准备训练数据的部分(collect)，通常包括模型前向推理生成数据的计算图(forward)，前后的数据处理，如何得到某个时间步的一个数据帧(transition)，如何把收集到的诸多数据帧组织成训练所用的样本(get_train_sample)
 - 算法进行模型性能评测的部分(eval)，通常包括模型前向推理进行评测的计算图(forward)，前后的数据处理
 - 算法在上述三个模块之间传递信息，控制和指挥三个模块的部分(command)，比如DQN中epsilon_greedy需要根据训练的迭代数来确定探索所用的epsilon值，并把这个值交给数据收集部分

而算法的其他结构，如：

 - ``Replay Buffer``
 - ``Env``

则由入口类serial_pipeline创建完成，不需要在Policy类中再进行实现

所有Policy需要继承 ``Policy`` 类，一些常用的方法实现被封装在 ``CommonPolicy`` 类中，如果不需要定制化也可以直接继承它

.. code:: python

    #所有Policy需要继承Policy类
    class DQNPolicy(CommonPolicy):
        r"""
        Overview:
            Policy class of DQN algorithm.
        """

我们需要对learn部分进行初始化，包括：

- 初始化learn的optimizer， 即 ``self._optimizer`` 
- 初始化算法的相关参数 
- 初始化learn所用的运行时模块learner armor ，即 ``self._armor``
- 初始化armor的相关model和plugin 

  - 如初始化target network(double dqn中的设计)

  - 如在训练时使用 ``argmax`` 进行sample 

为此我们实现 ``_init_learn`` 方法

.. code:: python

        
        def _init_learn(self) -> None:
            r"""
            Overview:
                Learn mode init method. Called by ``self.__init__``.
                Init the optimizer, algorithm config, main and target armors.
            """
            # Optimizer
            # 初始化learn的optimizer
            self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
    
            # Algorithm config
            # 初始化算法的相关参数
            algo_cfg = self._cfg.learn.algo
            self._nstep = algo_cfg.nstep
            self._gamma = algo_cfg.discount_factor
        
            # Main and target armors
            # 初始化的模型传入armor
            self._armor = Armor(self._model)
            
            # 初始化armor的相关model
            self._armor.add_model('target', update_type='assign', update_kwargs={'freq': algo_cfg.target_update_freq})
            # 初始化armor的相关plugin
            self._armor.add_plugin('main', 'argmax_sample')
            self._armor.add_plugin('main', 'grad', enable_grad=True)
            self._armor.add_plugin('target', 'grad', enable_grad=False)
            
            #常规初始化
            self._armor.mode(train=True)
            self._armor.target_mode(train=True)
            
            self._armor.reset()
            self._armor.target_reset()
            self._learn_setting_set = {}

我们的learner需要知道如何计算loss，并进行模型的更新等操作

为此我们实现 ``_forward_learn`` 方法

.. code:: python

    
        def _forward_learn(self, data: dict) -> Dict[str, Any]:
            r"""
            Overview:
                Forward and backward function of learn mode.
            Arguments:
                - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
            Returns:
                - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
            """
    
            # ====================
            # Q-learning forward
            # ====================
            # Reward reshaping for n-step
            reward = data['reward']
            if len(reward.shape) == 1:
                reward = reward.unsqueeze(1)
            assert reward.shape == (self._cfg.learn.batch_size, self._nstep), reward.shape
            reward = reward.permute(1, 0).contiguous()
            # Current q value (main armor)
            q_value = self._armor.forward(data['obs'])['logit']
            # Target q value
            target_q_value = self._armor.target_forward(data['next_obs'])['logit']
            # Max q value action (main armor)
            target_q_action = self._armor.forward(data['next_obs'])['action']
    
            data_n = q_nstep_td_data(
                q_value, target_q_value, data['action'], target_q_action, reward, data['done'], data['weight']
            )
            loss = q_nstep_td_error(data_n, self._gamma, nstep=self._nstep)
    
            # ====================
            # Q-learning update
            # ====================
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
    
            # =============
            # after update
            # =============
            self._armor.target_update(self._armor.state_dict()['model'])
            return {
                'cur_lr': self._optimizer.defaults['lr'],
                'total_loss': loss.item(),
            }

我们也需要对actor部分进行初始化，包括： 

- actor数据的收集方式， 包括 ``self._adder`` 等
- 初始化的模型传入actor armor， 即 ``self._collect_armor`` 
- 初始化armor的相关plugin 

  - 如actor使用 ``eps_greedy`` 进行sample

为此我们实现 ``_init_collect`` 方法

.. code:: python

      def _init_collect(self) -> None:
            r"""
            Overview:
                Collect mode init method. Called by ``self.__init__``.
                Init traj and unroll length, adder, collect armor.
                Enable the eps_greedy_sample
            """
            # actor数据的收集方式
            self._traj_len = self._cfg.collect.traj_len
            if self._traj_len == "inf":
                self._traj_len == float("inf")
            self._unroll_len = self._cfg.collect.unroll_len
            self._adder = Adder(self._use_cuda, self._unroll_len)
            self._collect_nstep = self._cfg.collect.algo.nstep
            
            # 初始化的模型传入actor armor
            self._collect_armor = Armor(self._model)
            
            # 初始化armor的相关plugin
            self._collect_armor.add_plugin('main', 'eps_greedy_sample')
            self._collect_armor.add_plugin('main', 'grad', enable_grad=False)
            
            # 常规初始化
            self._collect_armor.mode(train=False)
            self._collect_armor.reset()
            self._collect_setting_set = {'eps'}

我们的actor需要根据环境返回的observation获取相关动作数据

为此我们实现 ``_forward_collect`` 方法

.. code:: python

    
        def _forward_collect(self, data_id: List[int], data: dict) -> dict:
            r"""
            Overview:
                Forward function for collect mode with eps_greedy
            Arguments:
                - data_id (:obj:`List` of :obj:`int`): Not used, set in arguments for consistency
                - data (:obj:`dict`): Dict type data, including at least ['obs'].
            Returns:
                - data (:obj:`dict`): The collected data
            """
            return self._collect_armor.forward(data, eps=self._eps)

我们需要从trajectory（一组数据帧(transition)）中获取需要的训练样本

为此我们实现 ``_get_train_sample`` 方法

.. code:: python

    
        def _get_train_sample(self, traj_cache: deque) -> Union[None, List[Any]]:
            r"""
            Overview:
                Get the trajectory and the n step return data, then sample from the n_step return data
            Arguments:
                - traj_cache (:obj:`deque`): The trajectory's cache
            Returns:
                - samples (:obj:`dict`): The training samples generated
            """
            # adder is defined in _init_collect
            return_num = 0 if self._collect_nstep == 1 else self._collect_nstep
            data = self._adder.get_traj(traj_cache, self._traj_len, return_num=return_num)
            data = self._adder.get_nstep_return_data(data, self._collect_nstep, self._traj_len)
            return self._adder.get_train_sample(data)


我们需要将对应的数据加入transition，即在 ``BaseSerialActor`` 中实现的：

.. code:: python

    transition = self._policy.process_transition(self._obs_pool[env_id], self._policy_output_pool[env_id], timestep)



为此我们实现 ``_process_transition`` 方法，即获取一个时间步的数据帧(transition)

.. code:: python

    
        def _process_transition(self, obs: Any, armor_output: dict, timestep: namedtuple) -> dict:
            r"""
           Overview:
               Generate dict type transition data from inputs.
           Arguments:
               - obs (:obj:`Any`): Env observation
               - armor_output (:obj:`dict`): Output of collect armor, including at least ['action']
               - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                   (here 'obs' indicates obs after env step).
           Returns:
               - transition (:obj:`dict`): Dict type transition data.
           """
            transition = {
                'obs': obs,
                'next_obs': timestep.obs,
                'action': armor_output['action'],
                'reward': timestep.reward,
                'done': timestep.done,
            }
            return EasyDict(transition)


我们需要对evaluator部分进行初始化，包括：

-  初始化的模型传入 eval armor， 即 ``self._eval_armor``
-  初始化armor的相关plugin

   -  如使用 ``argmax`` 进行sample

为此我们实现 ``_init_eval`` 方法

我们的evaluator需要根据环境返回的observation获取相关动作数据

为此我们实现 ``_forward_eval`` 方法

.. code:: python

    
    
        def _init_eval(self) -> None:
            r"""
            Overview:
                Evaluate mode init method. Called by ``self.__init__``.
                Init eval armor with argmax strategy.
            """
            self._eval_armor = Armor(self._model)
            self._eval_armor.add_plugin('main', 'argmax_sample')
            self._eval_armor.add_plugin('main', 'grad', enable_grad=False)
            self._eval_armor.mode(train=False)
            self._eval_armor.reset()
            self._eval_setting_set = {}
    
        def _forward_eval(self, data_id: List[int], data: dict) -> dict:
            r"""
            Overview:
                Forward function for eval mode, similar to ``self._forward_collect``.
            Arguments:
                - data_id (:obj:`List[int]`): Not used in this policy.
                - data (:obj:`dict`): Dict type data, including at least ['obs'].
            Returns:
                - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
            """
            return self._eval_armor.forward(data)

在 ``_init_command`` 方法中，我们需要对相关控制模块进行初始化，比如epsilon_greedy的计算模块，使用者无需考虑信息在learner和actor之间如何传递，只需要考虑拿到信息后做怎样的数据处理即可

在 ``_get_setting_collect`` 方法中，我们使用command中的组件，根据相关信息（比如训练迭代数），来为collect部分设置下一次工作的相关配置

.. code:: python

    
        def _init_command(self) -> None:
            r"""
            Overview:
                Command mode init method. Called by ``self.__init__``.
                Set the eps_greedy rule according to the config for command
            """
            eps_cfg = self._cfg.command.eps
            self.epsilon_greedy = epsilon_greedy(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)
    
        def _get_setting_collect(self, command_info: dict) -> dict:
            r"""
            Overview:
                Collect mode setting information including eps
            Arguments:
                - command_info (:obj:`dict`): Dict type, including at least ['learner_step']
            Returns:
               - collect_setting (:obj:`dict`): Including eps in collect mode.
            """
            learner_step = command_info['learner_step']
            return {'eps': self.epsilon_greedy(learner_step)}
        
        

我们需要根据config初始化我们的模型，传给 ``self._model``

为此我们实现 ``_create_model_from_cfg`` 方法，该方法中会使用默认的神经网络模型，如果用户需要定制自己的神经网络，可通过 ``model_type`` 参数来处理。

.. code:: python

    
        def _create_model_from_cfg(self, cfg: dict, model_type: Optional[type] = None) -> torch.nn.Module:
            r"""
           Overview:
               Create a model according to input config. This policy will adopt DiscreteNet.
           Arguments:
               - cfg (:obj:`dict`): Config.
               - model_type (:obj:`Optional[type]`): If this is not None, this function will create \
                   an instance of this.
           Returns:
               - model (:obj:`torch.nn.Module`): Generated model.
           """
            if model_type is None:
                return FCDiscreteNet(**cfg.model)
            else:
                return model_type(**cfg.model)
    
    


在实现了Policy class之后，我们需要对该Policy class进行注册，这样 ``serial_pipeline`` 才能知道此policy的存在

.. code:: python

    # 注册dqn policy
    register_policy('dqn', DQNPolicy)


这样，我们就完成了 ``DQNPolicy`` 类的撰写


.. note::

    将上述各个模块组装起来构成完整的训练代码，nerveX提供了简易的 ``serial_pipeline`` ，可以参见 ``nervex/entry/serial_entry.py`` ，使用者
    需要编写 ``config`` 文件来完成自己的训练入口，具体可以参考 ``nervex/entry/tests/test_serial_entry.py`` 中的使用方式。
    
    此外，使用者还可以重写修改其他方法实现自定义功能。

以上指南简述了如何基于nerveX搭建一个最简单的DRL训练pipeline，训练配置文件各个字段
的具体含义则可以参见 `cartpole_dqn_cfg <../configuration/index.html#cartpole-dqn-config>`_，有其他的使用问题也可以参考文档的QA部分。
