How to understand training generated folders?
================================================================

DI-engine generates many folders during training: 

    - In **serial** mode, DI-engine generates log and checkpoint folders.
    - In **parallel** mode, DI-engine generates log, checkpoint, data and policy folders.

We will introduce these two modes one by one.

Serial mode
--------------------

In serial mode, generated file tree is as follows:

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

    In buffer folder, there is a file named ``buffer_logger.txt`` including some information about the data usage in the buffer.

    After a certain number of sample times, sample information will be printed to display the attributes of the sampled data, which demonstrating data quality. The table is like this:

    +-------+----------+----------+--------------+--------------+--------------+---------------+---------------+
    | Name  | use_avg  | use_max  | priority_avg | priority_max | priority_min | staleness_avg | staleness_max |
    +-------+----------+----------+--------------+--------------+--------------+---------------+---------------+
    | Value | float    | int      | float        | float        | float        | float         | float         |
    +-------+----------+----------+--------------+--------------+--------------+---------------+---------------+

    After a certain number of seconds, throughput information(number of push, sample, remove, valid) will be printed like this:

    +-------+--------------+--------------+--------------+--------------+
    | Name  | pushed_in    | sampled_out  | removed      | current_have |
    +-------+--------------+--------------+--------------+--------------+
    | Value | float        | float        | float        | float        |
    +-------+--------------+--------------+--------------+--------------+


- log/collector

    In collector folder, there is a file named ``collector_logger.txt`` including some information about the interaction with the environment.

    - Set default n_sample mode. The collector's basic information: n_sample and env_num. n_sample means the number of data samples collected. For env_num, it means how many environments the collector will interact with.
    

    - Special information when the collector interact with the environment,such as

        - episode_count: The count of collecting data episode
        - envstep_count: The count of collecting data envstep
        - train_sample_count: The count of train sample data 
        - avg_envstep_per_episode: Average envstep per eposide
        - avg_sample_per_episode: Average sample num per eposide
        - avg_envstep_per_sec: Average envstep per second
        - avg_train_sample_per_sec: Average train sample per second
        - avg_episode_per_sec: Average eposide per second
        - collect_time: How much time did the collector spend
        - reward_mean: Average reward
        - reward_std: The reward's standard deviation
        - each_reward: Each reward when the collector interact with an environment.
        - reward_max: The max reward
        - reward_min: The min reward
        - total_envstep_count: Total envstep number
        - total_train_sample_count: Total train sample number
        - total_episode_count: Total episode number
        - total_duration: Total duration

- log/evaluator

    In evaluator folder, there is a file named ``evaluator_logger.txt`` including some information about the evaluator when collector interacts with the environment.

    - [INFO]: env finish episode, final reward: xxx, current episode: xxx

    - train_iter: The train iter
    - ckpt_name: The model path, such as iteration_0.pth.tar
    - episode_count: The count of episode
    - envstep_count: The count of envstep
    - evaluate_time: How much time did the evaluator spend
    - avg_envstep_per_episode: Average envstep per eposide
    - avg_envstep_per_sec: Average envstep per second
    - avg_time_per_episode: Average time per eposide
    - reward_mean: Average reward
    - reward_std: The reward's standard deviation
    - each_reward: Each reward when the evaluator interact with an environment.
    - reward_max: The max reward
    - reward_min: The min reward


- log/learner

    In learner folder, there is a file named ``learner_logger.txt`` including some information about the learner.

    The following information is generated during DQN training

    - policy neural network architecture:

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



    - learner information:

        Grid table:

            +-------+------------+----------------+
            | Name  | cur_lr_avg | total_loss_avg |
            +-------+------------+----------------+
            | Value | 0.001000   | 0.098996       |
            +-------+------------+----------------+


- serial

    Save the related information of buffer, collector, evaluator, learner, to a file named ``events.out.tfevents``, and it can be used by **tensorboard**.
    
    DI-engine saves all tensorboard files in serial folder as **one tensorboard file**, rather than respective folders. Because when running a lot of experiments, 4*n respective tensorboard files is not easy to discriminate. So in serial mode, all tensorboard files are in the serial folder. (However, in parallel mode, tensorboard files are in respective folder)

- ckpt_baseLearner

    In this folder, there are model parameter checkpoints:
        - ckpt_best.pth.tar. Best model which reached highest evaluation score. 
        - "iteration" + iter number. Periodic model save. 

    You can use ``torch.load('ckpt_best.pth.tar')`` to load checkpoint.

Parallel mode
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



In parallel mode, the log folder has five subfolders, including buffer, collector, evaluator, learner, commander and a file coordinator_logger.txt

- log/buffer

    In buffer folder, there is a file named ``buffer_logger.txt`` and a subfolder named buffer_tb_logger.

    The data in ``buffer_logger.txt`` is the same as that in serial mode.

    In buffer_tb_logger folder, there is a ``events.out.tfevents`` tensorboard file.

- log/collector

    In collector folder, there are a lot of ``collector_logger.txt`` files including informations about the collector when collector interacts with the environment. There are a lot of collectors in parallel mode, so there are a lot of ``collector_logger.txt`` files record informations.

    The data in ``collector_logger.txt`` is the same as serial mode.

- log/evaluator

    In evaluator folder, there are a lot of ``evaluator_logger.txt`` files including informations about the evaluator when evaluator interacts with the environment. There are a lot of evaluators in parallel mode, so there are a lot of ``evaluator_logger.txt`` files record informations.

    The data in ``evaluator_logger.txt`` is the same as serial mode.

- log/learner

    In learner folder, there is a file named ``learner_logger.txt`` and a subfolder named learner_tb_logger.

    The data in ``learner_logger.txt`` is the same as serial mode.

    In learner_tb_logger folder, there are some files ``events.out.tfevents``, and it can be used by tensorboard.

    In parallel mode, it's too difficult to put all tb files in the same folder, so each tb file is placed in a folder with its corresponding text logger file. It's different from th eserial mode. In serial mode, we put all tb files in serial folder.

- log/commander

    In commander folder, there are three files: ``commander_collector_logger.txt``, ``commander_evaluator_logger.txt``, ``commander_logger.txt`` and a subfolder named learner_tb_logger.

    In ``commander_collector_logger.txt``, there are some collector's information the coordinator needs. Such as train_iter, step_count, avg_step_per_episode, avg_time_per_step, avg_time_per_episode, reward_mean, reward_std

    In ``commander_evaluator_logger.txt``, there are some evaluator's information the coordinator needs. Such as train_iter, step_count, avg_step_per_episode, avg_time_per_step, avg_time_per_episode, reward_mean, reward_std

    In ``commander_logger.txt``, there are some information when the coordinator will be end.

    There are so many files in the collector and evaluator folder that it seems inconvenient. So we made a synthesis in the commander. This is the reason why there are collector and evaluator folders in parallel mode but the commander folder has collector text file and evaluator text file.


- ckpt:

    Parallel mode's checkpoint folder is the same as serial mode's.

    In this folder, there are model parameter checkpoints:
        - ckpt_best.pth.tar. Best model which reached highest evaluation score. 
        - "iteration" + iter number. Periodic model save. 

    You can use ``torch.load('ckpt_best.pth.tar')`` to load checkpoint.


- data

    In this folder, there are a lot of data files. In serial mode, all datas are stored in memory; While in parallel mode, data is separated into meta data and file data: meta data is still stored in memory, but file data is stored in file system.

- policy

    In this folder, there is a policy file. The file includes policy parameters. It is used to send learner's latest parameters to collector to update. In parallel mode, the coordinator uses the path of the policy file to register the collector, the collector uses data in policy file as its own parameters.
