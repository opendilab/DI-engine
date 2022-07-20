如何理解训练过程中生成的文件夹？
================================================================

DI-engine 在训练过程中会生成很多文件夹：

    - 在 **串行（serial）** 模式下, DI-engine 生成 log 和 checkpoint 文件夹.
    - 在 **并行（parallel）** 模式下, DI-engine 生成 log, checkpoint, data 和 policy 文件夹.

我们将分别介绍这两种模式。

串行模式
--------------------

在串行模式下，生成的文件树如下：

::

    cartpole_dqn
    ├── ckpt
    │   ├── ckpt_best.pth.tar
    │   ├── iteration_0.pth.tar
    │   └── iteration_561.pth.tar
    ├── formatted_total_config.py
    ├── log
    │   ├── buffer
    │   │   └── buffer_logger.txt
    │   ├── collector
    │   │   └── collector_logger.txt
    │   ├── evaluator
    │   │   └── evaluator_logger.txt
    │   ├── learner
    │   │   └── learner_logger.txt
    │   └── serial
    │       └── events.out.tfevents.1626453528.CN0014009700M.local
    └── total_config.py


- log/buffer

    在 buffer 这个文件夹里, 有一个名为 ``buffer_logger.txt`` 的文件，其中包含buffer中有关数据使用情况的一些信息。

    经过一定次数的采样后，会打印出采样信息，显示采样数据的属性，并且也展示出数据的质量。 表格如下图所示：

    +-------+----------+----------+--------------+--------------+--------------+---------------+---------------+
    | Name  | use_avg  | use_max  | priority_avg | priority_max | priority_min | staleness_avg | staleness_max |
    +-------+----------+----------+--------------+--------------+--------------+---------------+---------------+
    | Value | float    | int      | float        | float        | float        | float         | float         |
    +-------+----------+----------+--------------+--------------+--------------+---------------+---------------+

    一定秒数后，吞吐量信息（进入buffer的数据数量、采样次数、删除数量、目前拥有的有效数量）将打印如下：

    +-------+--------------+--------------+--------------+--------------+
    | Name  | pushed_in    | sampled_out  | removed      | current_have |
    +-------+--------------+--------------+--------------+--------------+
    | Value | float        | float        | float        | float        |
    +-------+--------------+--------------+--------------+--------------+


- log/collector

    在收集器文件夹中，有一个名为“collector_logger.txt”的文件，其中包含一些与环境交互相关的信息。

    - 设默认置 n_sample 模式。 collector 的基本信息: n_sample 和 env_num. n_sample 表示采集的数据样本数. 对于 env_num，它表示collector将与多少个环境交互。
    

    - collector与环境交互时产生的特殊信息，例如

        - episode_count: 收集数据的episode数量
        - envstep_count: 收集数据的envstep数量
        - train_sample_count: 训练样本数据个数
        - avg_envstep_per_episode: 每个 eposide中平均的 envstep
        - avg_sample_per_episode: 每个episode中的平均样本数
        - avg_envstep_per_sec: 每秒平均的env_step
        - avg_train_sample_per_sec: 每秒平均的训练样本数
        - avg_episode_per_sec: 每秒平均episode数
        - collect_time: 收集时间
        - reward_mean: 平均奖励
        - reward_std: 奖励的标准差
        - each_reward: collector与环境交互时的每个episode的奖励。
        - reward_max: 最大reward
        - reward_min: 最小reward
        - total_envstep_count: 总 envstep 数
        - total_train_sample_count: 总训练样本数
        - total_episode_count: 总 episode 数
        - total_duration: 总持续时间


- log/evaluator

    在 evaluator 文件夹中，有一个名为 ``evaluator_logger.txt`` 的文件，其中包含有关 evaluator 与环境交互时的一些信息。

    - [INFO]: env 完成episode，最终奖励：xxx，当前episode：xxx

    - train_iter: 训练迭代数
    - ckpt_name: 模型路径，如iteration_0.pth.tar
    - episode_count: episode计数
    - envstep_count: envstep计数
    - evaluate_time: evaluator花费的时间
    - avg_envstep_per_episode: 每个episode的平均envstep
    - avg_envstep_per_sec: 每秒的平均envstep
    - avg_time_per_episode: 每秒的平均episode
    - reward_mean: 平均奖励
    - reward_std: 奖励的标准差
    - each_reward: evaluator与环境交互时的每个episode的奖励。
    - reward_max: 最大reward
    - reward_min: 最小reward


- log/learner

    在learner文件夹中，有一个名为“learner_logger.txt”的文件，其中包含有关learner的一些信息。

    以下信息是在 DQN 训练期间生成的

    - 策略神经网络架构：
        ::

            INFO:learner_logger:[RANK0]: DI-engine DRL Policy
            DQN(
              (encoder): FCEncoder(
                (act): ReLU()
                (init): Linear(in_features=4, out_features=128, bias=True)
                (main): Sequential(
                  (0): Linear(in_features=128, out_features=128, bias=True)
                  (1): ReLU()
                  (2): Linear(in_features=128, out_features=64, bias=True)
                  (3): ReLU()
                )
              )
              (head): DuelingHead(
                (A): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=64, out_features=64, bias=True)
                    (1): ReLU()
                  )
                  (1): Sequential(
                    (0): Linear(in_features=64, out_features=2, bias=True)
                  )
                )
                (V): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=64, out_features=64, bias=True)
                    (1): ReLU()
                  )
                  (1): Sequential(
                    (0): Linear(in_features=64, out_features=1, bias=True)
                  )
                )
              )
            )



    - leaner信息：

        网格表：

            +-------+------------+----------------+
            | Name  | cur_lr_avg | total_loss_avg |
            +-------+------------+----------------+
            | Value | 0.001000   | 0.098996       |
            +-------+------------+----------------+


- serial

    将buffer、collector、evaluator、learner的相关信息保存到名为 ``events.out.tfevents`` 的文件中，供 **tensorboard** 使用。
    
    DI-engine 将串行文件夹中的所有 tensorboard 文件保存为 **一个 tensorboard 文件 ** ，而不是各自的文件夹。 因为在跑如果跑n个实验的时候，当n很大时，4*n个各自的tensorboard文件不容易判别。 所以在串行模式下，所有的 tensorboard文件都在串行文件夹中 （但是，在并行模式下，tensorboard文件位于各自的文件夹中）。

- ckpt

    在这个文件夹中，有模型参数 checkpoints：
        - ckpt_best.pth.tar. 达到最高评价分数的最佳模型. 
        - "iteration" + iter number. 每 ``iter_number`` 保存的模型。 

    您可以使用 ``torch.load('ckpt_best.pth.tar')`` 来加载模型。

并行模式
--------------------

::

    cartpole_dqn
    ├── ckpt
    │   └── iteration_0.pth.tar
    ├── data
    ├── log
    │   ├── buffer
    │   │   ├── buffer_logger.txt
    │   │   └── buffer_tb_logger
    │   │       └── events.out.tfevents.1626453752.CN0014009700M.local
    │   ├── collector
    │   │   ├── 4890b4c5-f084-4c94-b440-75f9fa602388_614285_logger.txt
    │   │   ├── c029d882-fe4f-4a1d-9451-13015bbca192_750418_logger.txt
    │   │   └── fc68e215-f062-4a1b-a0fd-dcf5f375b290_886803_logger.txt
    │   ├── commander
    │   │   ├── commander_collector_logger.txt
    │   │   ├── commander_evaluator_logger.txt
    │   │   ├── commander_logger.txt
    │   │   └── commander_tb_logger
    │   │       └── events.out.tfevents.1626453748.CN0014009700M.local
    │   ├── coordinator_logger.txt
    │   ├── evaluator
    │   │   ├── 1496df45-8858-4f38-82da-b4a39461a268_451909_logger.txt
    │   │   └── 2e8879e3-8af5-4ebb-8d50-8af829f03845_711157_logger.txt
    │   └── learner
    │       ├── learner_logger.txt
    │       └── learner_tb_logger
    │           └── events.out.tfevents.1626453750.CN0014009700M.local
    └── policy
        ├── policy_0d2a6a81-fd73-4e29-8815-3607f1428aaa_907961
        └── policy_0d2a6a81-fd73-4e29-8815-3607f1428aaa_907961.lock:



并行模式下，log文件夹有5个子文件夹，包括buffer、collector、evaluator、learner、commander和一个文件coordinator_logger.txt

- log/buffer

    在 buffer 文件夹中, 有一个名为 ``buffer_logger.txt`` 的文件和一个名为 ``buffer_tb_logger`` 的子文件夹。

    ``buffer_logger.txt`` 中的数据与串行模式下的数据相同。

    在 buffer_tb_logger 文件夹中，有一个 ``events.out.tfevents`` tensorboard 文件。

- log/collector

    在collector文件夹中，有很多 ``collector_logger.txt`` 文件，包括collector与环境交互时的collector信息。并行模式有很多collector，所以有很多 ``collector_logger.txt`` 文件记录信息。

    ``collector_logger.txt``中的数据与串行模式相同。

- log/evaluator

    在 evaluator 文件夹中，有很多 ``evaluator_logger.txt`` 文件，包括 evaluator 与环境交互时有关 evaluator 的信息。 并行模式有很多evaluator，所以有很多 ``evaluator_logger.txt`` 文件记录信息。

    ``evaluator_logger.txt`` 中的数据与串行模式相同。

- log/learner

    在learner文件夹中，有一个名为 ``learner_logger.txt`` 的文件和一个名为 ``learner_tb_logger`` 的子文件夹。

    ``learner_logger.txt`` 中的数据与串行模式相同。

    在 learner_tb_logger 文件夹中保存了一些 tensorboard 文件 ``events.out.tfevents``，可以被 tensorboard 使用。

    在并行模式下，将所有tb文件放在同一个文件夹中太难了, 所以每个 tb 文件都放在一个文件夹中，其中包含相应的文本记录器文件。 它与串行模式不同。在串行模式下，我们将所有 tb 文件放在串行文件夹中。

- log/commander

    在commander文件夹中，有三个文件： ``commander_collector_logger.txt``, ``commander_evaluator_logger.txt``, ``commander_logger.txt`` 和一个名为 ``learner_tb_logger`` 的子文件夹.

    在 ``commander_collector_logger.txt`` 里, 有一些coordinator需要的collector的信息。 如train_iter、step_count、avg_step_per_episode、avg_time_per_step、avg_time_per_episode、reward_mean、reward_std

    在 ``commander_evaluator_logger.txt`` 里, 有一些coordinator需要的evaluator的信息。 如train_iter、step_count、avg_step_per_episode、avg_time_per_step、avg_time_per_episode、reward_mean、reward_std

    在 ``commander_logger.txt`` 里, 有一些关于coordinator何时将会结束的信息

    collector和evaluator文件夹中有很多文件，看起来很不方便。 所以我们在commander里面做了一个整合。 这就是并行模式下存在collector 和evaluator 文件夹但commander 文件夹具有collector 文本文件和evaluator 文本文件的原因。


- ckpt:

    并行模式下 checkpoint 文件夹与串行模式的相同。

    在这个文件夹中，有模型参数 checkpoints：
        - ckpt_best.pth.tar. 达到最高评价分数的最佳模型. 
        - "iteration" + iter number. 每 ``iter_number`` 保存的模型。 

    您可以使用 ``torch.load('ckpt_best.pth.tar')`` 来加载模型。


- data

    在这个文件夹中，有很多数据文件。 在串行模式下，所有数据都存储在内存中; 在并行模式下，数据分为元数据和文件数据： 元数据仍然存储在内存中，但文件数据存储在文件系统中。

- policy

    在此文件夹中，有一个策略文件。 该文件包含策略参数。用于将learner的最新参数发送给collector进行更新。 在并行模式下，coordinator使用策略文件的路径注册collector，collector使用策略文件中的数据作为自己的参数。
