Actor Overview
================
概述：
    Actor是为训练学习端提供足够数量和质量数据的模块，但Evaluator也作为一种不存储和传递时间步数据的特殊Actor存在。主要包括三大模块:

        - Actor Controller(数据生成控制器)
        - Env Manager(环境管理器)
        - Armor(智能体)


Actor Controller(数据生成控制器)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
概述：
    Actor Controller(以下简称AC), 是Actor部分的管理模块和主入口，需要研究员根据自己的实际需求，继承 ``BaseActor`` 基类实现相关接口。该模块维护Actor的基本信息和与Coordinator的通信，常驻在某个机器上。

代码结构：
    - ``base_actor_controller`` ： actor controller基类，定义基础接口方法，主循环可参考 ``run`` 方法。actor以job为工作的基本单位，coordinator设置job内容和所需计算资源。AC得到job后，建立其和armor(模型推理)以及env_manager(环境模拟) 之间的联系，并根据job执行执行一个或多个episode。AC会会维护单独的线程定期异步地更新armor
      。对于数据，当累积的数据量满足一定要求后（例如一定长度），AC会将这部分数据进行打包发送会coordinator。当某个job全部运行完毕后，AC也会将相应信息返回给coordinator。
    - ``comm`` 通信模块：该部分被隐式地封装，通过python的动态属性机制绑定在AC上，研究员只需在配置文件中指定相应的选项即可，在AC的实现代码中只需调用具体的通信接口，而无需涉及具体的通信过程，如果对具体的通信过程感兴趣，可以详细阅读 ``comm`` 部分的相关代码

功能特性：
    - AC启动后就建立和coordinator之间的通信（维护一个心跳线程），但具体的armor资源(GPU)，env资源(CPU)还是由coordinator负责管理，AC根据job进行相应处理，这样支持不同job使用不同的资源。
    - AC对于armor的更新是异步的，一般是固定时间间隔进行一次相关信息加载和更新。但对于可能出现的即时更新需求，之后也可提供相应的接口 (TODO)。
    - AC主循环接近标准最基础的RL交互迭代过程，即从环境获得state->模型获得action->环境获得obs_next和reward的迭代，具体的定制化需求可在各接口方法中实现。

      .. image:: rl_iter.png

    - AC使用向量化env(env_manager)和batch inference机制来进行效率优化，故对于数据的打包等操作也在AC中完成。但注意可能存在大量的数据不等长情况（例如向量化运行8个env但各个env的结束时间不同，致使在某些状态下会只有部分env执行交互），这时候需要在打包阶段进行相应处理。
    - AC维护actor相关的各类日志信息。
    - 整个Actor部分一般运行在单机上，各个组件之间一般使用IPC进行通信，之后会研究如果利用单机共享内存来避免多余的数据拷贝(TODO)，即AC中只获得共享内存中数据的引用来进行管理。


Communication Actor(数据生成器通信模块)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
概述：
    数据通信主要包含三类需求：

        - actor和coordinator的信息通信，包含actor状态更新（心跳线程），job获取，job完成信息返回等
        - armor更新，一般是actor从读取相应信息（比如神经网络模型）来更新armor
        - 生成数据的发送，数据以trajectory为基本单位，即累积一定长度就进行发送。这里采用metadata和stepdata分离的机制，即将metadata返回coordinator，而将stepdata存入某些数据容器（比如ceph or redis）

    目前支持的通信方式有：
        
        - flask-file_system：即通过flask框架，走网络通信完成actor和coordinator的交互，而大块数据的读取则通过文件系统（现在支持一般的磁盘读取和ceph）


Env Manager(环境管理器)
~~~~~~~~~~~~~~~~~~~~~~~~~
概述：
    env manager是一个向量化的环境管理器，其中同时运行多个相同类型不同配置的环境，实际实现方式包含子进程向量化和伪向量化（循环串行）两种模式，具体可参考 `env_manager_overview <../env_manager/env_manager_overview.html>`_ 。

Armor(智能体)
~~~~~~~~~~~~~~

概述：
    armor作为runtime的算法模型，支持运行时的各种动态功能，具体可参考 `armor_overview <../armor/armor_overview.html>`_。当其作为actor的一部分时，主要支持batch inference和指定样本id的inference。
