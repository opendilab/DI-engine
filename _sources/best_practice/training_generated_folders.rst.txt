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

    ./
    ├── log
    │   ├── buffer
    │   │   └── agent_buffer_logger.txt
    │   ├── collector
    │   │   └── collect_logger.txt
    │   ├── evaluator
    │   │   └──  evaluator_logger.txt
    │   ├── learner
    │   │   └── learner_logger.txt
    │   └── serial
    └── ckpt_baseLearner_ (ckpt_BaseLearner_Mon_May_24_12_08_43_2021)
        └── ckpt_best.pth.tar


- log/buffer

    In buffer folder, there is a file named ``agent_buffer_logger.txt`` including some information about the data usage in the buffer.

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

    The following information is generated during PPO training

    - learner config and model:

        ::

            config:
                cfg_type: BaseLearnerDict
                dataloader:
                num_workers: 0
                hook:
                    load_ckpt_before_run: ''
                    log_show_after_iter: 100
                    save_ckpt_after_iter: 10000
                    save_ckpt_after_run: true
                train_iterations: 1000000000
            FCValueAC(
            (_act): ReLU()
            (_encoder): FCEncoder(
                (act): ReLU()
                (init): Linear(in_features=4, out_features=64, bias=True)
                (main): ResFCBlock(
                (act): ReLU()
                (fc1): Sequential(
                    (0): Linear(in_features=64, out_features=64, bias=True)
                    (1): ReLU()
                )
                (fc2): Sequential(
                    (0): Linear(in_features=64, out_features=64, bias=True)
                )
                )
            )
            (_actor): Sequential(
                (0): Linear(in_features=64, out_features=128, bias=True)
                (1): ReLU()
                (2): Linear(in_features=128, out_features=128, bias=True)
                (3): ReLU()
                (4): Linear(in_features=128, out_features=2, bias=True)
            )
            (_critic): Sequential(
                (0): Linear(in_features=64, out_features=128, bias=True)
                (1): ReLU()
                (2): Linear(in_features=128, out_features=128, bias=True)
                (3): ReLU()
                (4): Linear(in_features=128, out_features=1, bias=True)
            )
            )


    - learner information:

        Grid table:

        +-------+------------+----------------+-----------------+----------------+------------------+-----------------+---------------+--------------+
        | Name  | cur_lr_val | total_loss_val | policy_loss_val | value_loss_val | entropy_loss_val | adv_abs_max_val | approx_kl_val | clipfrac_val |
        +-------+------------+----------------+-----------------+----------------+------------------+-----------------+---------------+--------------+
        | Value | 0.001000   | -0.421546      | -4.209646       | 10.286912      | 0.691280         | 6.281444        | 0.000000      | 0.000000     |
        +-------+------------+----------------+-----------------+----------------+------------------+-----------------+---------------+--------------+

        +-------+----------------+------------+----------------+-----------------+----------------+------------------+-----------------+---------------+--------------+
        | Name  | train_time_val | cur_lr_val | total_loss_val | policy_loss_val | value_loss_val | entropy_loss_val | adv_abs_max_val | approx_kl_val | clipfrac_val |
        +-------+----------------+------------+----------------+-----------------+----------------+------------------+-----------------+---------------+--------------+
        | Value | 0.004722       | 0.001000   | -0.888706      | -4.184078       | 9.948707       | 0.686777         | 7.128615        | 0.005156      | 0.000000     |
        +-------+----------------+------------+----------------+-----------------+----------------+------------------+-----------------+---------------+--------------+


- serial

    Save the related information of buffer, collector, evaluator, learner, to a file named ``events.out.tfevents``, and it can be used by **tensorboard**.
    
    DI-engine saves all tensorboard files in serial folder as **one tensorboard file**, rather than respective folders. Because when running a lot of experiments, 4*n respective tensorboard files is not easy to discriminate. So in serial mode, all tensorboard files are in the serial folder. (However, in parallel mode, tensorboard files are in respective folder)

- ckpt_baseLearner

    The folder is named in the way of "ckpt_baseLearner" + creation time (e.g. ``"Mon_May_24_12_08_43_2021"``).

    In this folder, there are model parameter checkpoints:
        - ckpt_best.pth.tar. Best model which reached highest evaluation score. 
        - "iteration" + iter number. Periodic model save. 

    You can use ``torch.load('ckpt_best.pth.tar')`` to load checkpoint.

Parallel mode
--------------------

::

    ./
    ├── log
    │   ├── buffer
    │   │   ├── agent_buffer_tb_logger
    │   │   └── agent_buffer_logger.txt
    │   ├── collector
    │   │   ├── 3b5f970b-0ff0-4394-bf8a-de43cadfd2b6_196408_logger.txt
    │   │   ├── XXX_X_logger.txt
    │   │   └── ...
    │   ├── evaluator
    │   │   ├── 3e483ac6-4a6e-4787-bfef-08f7cc3f14b8_300574_logger.txt
    │   │   ├── XXX_X_logger.txt
    │   │   └── ...
    │   ├── learner
    │   │   ├── learner_tb_logger
    │   │   └── learner_logger.txt
    │   ├── commander
    │   │   ├── commander_tb_logger
    │   │   ├── commander_collector_logger.txt       
    │   │   ├── commander_evaluator_logger.txt
    │   │   └── commander_logger.txt
    │   └── coordinator_logger.txt
    ├── ckpt_baseLearner_ (ckpt_BaseLearner_Mon_May_24_12_08_43_2021)
    │   └── iteration_.pth.tar (iteration_1000.pth.tar)
    ├── data
    │   ├── env_ (env_0_0aa0d0b4-c20c-11eb-9cd2-dd796209c19b)
    │   └── env_ (env_0_0a9e0488-c20c-11eb-9cd2-dd796209c19b) ...
    └── policy
        └── policy_0ee6e602-9d10-4aff-84a3-980a726430f7_222729



In parallel mode, the log folder has five subfolders, including buffer, collector, evaluator, learner, commander and a file coordinator_logger.txt

- log/buffer

    In buffer folder, there is a file named ``agent_buffer_logger.txt`` and a subfolder named agent_buffer_tb_logger.

    The data in ``agent_buffer_logger.txt`` is the same as that in serial mode.

    In agent_buffer_tb_logger folder, there is a ``events.out.tfevents`` tensorboard file.

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


- ckpt_baseLearner :

    Parallel mode's checkpoint folder is the same as serial mode's.

    The folder is named in the way of "ckpt_baseLearner" + creation time (e.g. ``"Mon_May_24_12_08_43_2021"``).

    In this folder, there are model parameter checkpoints:
        - ckpt_best.pth.tar. Best model which reached highest evaluation score. 
        - "iteration" + iter number. Periodic model save. 

    You can use ``torch.load('ckpt_best.pth.tar')`` to load checkpoint.


- data

    In this folder, there are a lot of data files. In serial mode, all datas are stored in memory; While in parallel mode, data is separated into meta data and file data: meta data is still stored in memory, but file data is stored in file system.

- policy

    In this folder, there is a policy file. The file includes policy parameters. It is used to send learner's latest parameters to collector to update. In parallel mode, the coordinator uses the path of the policy file to register the collector, the collector uses data in policy file as its own parameters.