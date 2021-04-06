# Serial Config说明

在某个具体环境具体算法的config文件中，必须包含以下key：``env`` ``policy`` ``replay_buffer`` ``actor`` ``evaluator`` ``learner`` ``commander``。

同时，nerveX对于部分配置项有**默认值**，例如：``replay_buffer``的默认配置在``nervex/config/buffer_manager.py``，``learner``的默认配置在``nervex/config/serial.py``。

下面，将依次对所有key进行介绍。在每个表格，还附带**Pong-DQN Serial Config的实例**，以供参考。

## env

| Name                      | Type                                 | Description                                                  |
| ------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| env.env_manager_type      | Enum[str]:``['base', 'subprocess']`` | **环境管理器**的类型。其中，`` 'base'``会串行执行各环境的step；而``'subprocess'``会为每个环境开启一个子进程，采用并行的方式执行各环境的step。 |
| env.manager               | Dict[str, Any]                       | 环境管理器的配置信息，目前仅在``env_manager_type``为``'subprocess'``时传递一些参数。通常包含以下key：``shared_memory`` ``context`` ``wait_num`` ``timeout``。 |
| env.manager.shared_memory | bool                                 | 是否为子进程们分配一块共享内存，用于缩短进程间通信的时间。   |
| env.manager.context       | str                                  | 开启子进程所用函数。默认情况下，Windows平台为``'spawn'``，其它平台为``'fork'``。 |
| env.manager.wait_num      | int                                  | ``'subprocess'``环境管理器的逻辑是：等待至少``wait_num``个环境``step``完毕，等待``timeout``秒，然后返回。可以根据环境运行速度的快慢来调整这两个参数。 |
| env.manager.timeout       | float                                | 见上。                                                       |
| env.import_names          | List[str]                            | 环境类的定义文件路径。需为``list``形式（即便只有一个需要import时也是如此），要求每个路径必须为**绝对路径**，即可以在python idle内执行``import name1.name2``，例如``['app_zoo.classic_control.cartpole.envs.cartpole_env']``。以下所有``import_names``都同理。其余路径，若不明确指明，均为**相对路径**即可。 |
| env.env_type              | Enum[str]: 具体请参考``app_zoo``     | 环境在注册时所使用的名字，例如``'cartpole'`` ``'pendulum'``  ``'atari'`` ``'mujoco'`` ``'smac'``等（若为``'atari'`` ``'mujoco'``这类集成了多个小环境的大环境，还需要包含 ``env_id``这一项，详见表格下一行）。 |
| env.env_id                | str                                  | 若``env_type``为``'atari'`` ``'mujoco'等``，**必须**在该项中指定小环境的名字。例如，``'atari'``中可为``'pong'`` ``'qbert'``等，``'mujoco'``中可为``'Ant-v3'`` ``'HalfCheetah-v3'``等。 |
| env.actor_env_num         | int                                  | 开启多少个环境供actor采集数据，必须为正整数。                |
| env.evaluator_env_num     | int                                  | 开启多少个环境供evaluator评测policy的效果，必须为正整数。    |
| env中其他可能出现的项     | Any                                  | Aatari环境中，可能还包括``frame_stack`` 等项。MuJoCo环境中，可能还包括``norm_obs`` ``norm_reward`` ``use_act_scale``等项。将在下一部分以例子的形式说明。 |

**实例**

```python
pong_dqn_default_config = dict(
    env=dict(
        env_manager_type='subprocess',
        import_names=['app_zoo.atari.envs.atari_env'],
        env_type='atari',
        env_id='PongNoFrameskip-v4',
        actor_env_num=16,
        evaluator_env_num=8,
        frame_stack=4,
    ),
    # ...
)
```

除了上述项之外，Atari pong环境中还有``frame_stack``项，表明会将多少帧堆叠在一起作为observation。



## policy

| Name                                 | Type                                           | Description                                                  |
| ------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------ |
| policy.use_cuda                      | bool                                           | policy中的model是否使用cuda。（目前默认learn、collect等阶段对于cuda的使用需求相同。） |
| policy.policy_type                   | Enum[str]:  具体请参考``nervex -q policy``命令 | policy在注册时使用的名字。若自己实现了policy且通过``Registry``机制进行了注册，也可在此处使用自己实现的policy的名字。 |
| policy.on_policy                     | bool                                           | 该策略是否为on-policy的策略。若为是，则不会在replay buffer中储存数据；若为否，则会将actor产生的数据先存入replay buffer，在learner需要训练时再从replay buffer中sample得到。 |
| policy.use_priority                  | bool                                           | 是否在replay buffer中使用优先级机制。若此项为``True``，必须在策略的对应方法中返回所需要的值。具体实现方法可参考Best Practice第一条。 |
| policy.model                         | Dict[str, Any]                                 | policy中使用何种神经网络进行inference。若计划在serial pipeline中传入参数``model``，那么以下包含哪些项则完全自定义；若计划使用policy的默认model，可直接参考其``default_model``方法。通常情况下，会包含以下key：``obs_dim`` ``action_dim``等。**特别提醒**：当更改环境时，一定要记得修改``obs_dim`` 和``action_dim``。 |
| policy.model.obs_dim                 | Union[int, List[int]]                          | 环境反馈的observation的维度。若有多个维度，需使用``list``。  |
| policy.model.action_dim              | Union[int, List[int]]                          | policy产生的action的维度。若有多个维度，需使用``list``。     |
| policy.model中其他可能出现的项       | Any                                            | 若使用自定义model，由于model类型过多，很难在此说明清楚，请直接参考``default_model``方法中指明的model所在的类。将在下一部分以例子的形式说明。 |
| policy.learn                         | Dict[str, Any]                                 | policy的learn模式中需要用到的参数，通常包含以下key：``train_iteration`` ``batch_size`` ``learning_rate`` ``weight_decay`` ``algo``。 |
| policy.learn.train_iteration              | int                                            | 在serial pipeline中，actor和learner交替工作。``train_iteration``是指，当轮到learner工作时，learner会调用``policy._forward_learn``的次数。必须为正整数。该数值越大，迭代越快，但使用的数据就越off-policy。 |
| policy.learn.batch_size              | int                                            | learner更新策略时，一个train iteration所用的batch包含多少个训练样本。 |
| policy.learn.learning_rate           | float                                          | learner更新网络参数时使用的学习率。                          |
| policy.learn.weight_decay            | float                                          | learner更新网络参数时使用的正则项系数。                      |
| policy.learn.algo                    | Dict[str, Any]                                 | policy的learn模式中和算法最直接相关的参数，通常包含以下key：``target_update_freq`` ``discount_factor``  。 |
| policy.learn.algo.target_update_freq | int                                            | 若使用target network，则可设置该值为其从main network复制网络参数以更新自身的频率。 |
| policy.learn.algo.discount_factor    | float                                          | 计算累计奖励（acummulative reward）时的折扣因子。必须为[0, 1]区间内的浮点数。 |
| policy.learn.algo中其他可能出现的项  | Dict[str, Any]                                 | 和具体算法强相关，建议直接查看对应policy。将在下一部分以例子的形式说明。 |
| policy.collect                       | Dict[str, Any]                                 | policy的learn模式中需要用到的参数，通常包含以下key：``traj_len`` ``unroll_len`` ``algo``。 |
| policy.collect.unroll_len            | int                                            | actor将收集到的轨迹切割成多段，每段长度为``unroll_len``，该项通常设置为1即可，除非模型或算法有要求（例如RNN模型）。 |
| policy.collect.algo                  | Dict[str, Any]                                 | policy的collect模式中和算法最直接相关的参数。                |
| policy.command                       | Dict[str, Any]                                 | policy的learn模式中需要用到的参数。例如若使用epsilon贪婪算法进行探索，则包含key：``eps`` 。 |
| policy.command.eps                   | Dict[str, Any]                                 | 包含以下四个key：``type``表示使用何种衰减方式，支持指数型``exp``和线型``linear``；``start``表示epsilon的初始值；``end``表示epsilon衰减的最小值；``decay``为衰减过程中用到的参数。 |

**实例**

```python
pong_dqn_default_config = dict(
	# ...
	policy=dict(
        use_cuda=True,
        policy_type='dqn',
        on_policy=False,
        use_priority=True,
        model=dict(
            obs_dim=[4, 84, 84],
            action_dim=6,
            encoder_kwargs=dict(encoder_type='conv2d', ),
            embedding_dim=512,
            head_kwargs=dict(dueling=False, ),
        ),
        learn=dict(
            train_iteration=20,
            batch_size=32,
            learning_rate=0.0001,
            weight_decay=0.0,
            algo=dict(
                target_update_freq=500,
                discount_factor=0.99,
                nstep=nstep,
            ),
        ),
        collect=dict(
            unroll_len=1,
            algo=dict(nstep=nstep, ),
        ),
        command=dict(eps=dict(
            type='exp',
            start=1.,
            end=0.05,
            decay=200000,
        ), ),
    # ...
)
```

对于``policy.model``：通过``DQNPolicy``的``default_model``方法，可以得知采用``FCDiscreteNet``作为默认的网络。``encoder_kwargs``包含encoder的配置信息，``embedding_dim``指明encoder与head连接处的维度数，``head_kwargs``包含head的配置信息。

对于``policy.learn.algo``：``nstep``指明使用多少步的td-error，代表在learner端，使用处理为nstep的数据进行训练。该项的默认值为1。相应地，在``policy.collect.algo.nstep``项通常与``policy.learn.algo.nstep``**保持一致**，代表在actor端，将原始轨迹数据处理为nstep的训练数据。



## replay_buffer

| Name                                       | Type           | Description                                                  |
| ------------------------------------------ | -------------- | ------------------------------------------------------------ |
| replay_buffer.replay_buffer_size     | int            | replay buffer的长度。                                         |
| replay_buffer.replay_start_size      | int            | replay buffer初始采样积累数据的数量。                         |
| replay_buffer.max_reuse              | int            | replay buffer中一个数据可以被使用的次数。当一个数据第``max_reuse``次被sample到时，将其从replay buffer中删除。 |
| replay_buffer.max_staleness          | int            | replay buffer中一个数据最大陈旧度。staleness被定义为actor端采集时的policy和learner端即将被优化的policy，二者iteration的差值。当对replay buffer进行sample时，先调用``sample_check``方法删去所有过于陈旧的数据，然后才能实际调用``sample``进行采样。 |
| replay_buffer.alpha                  | float          | replay buffer中优先级的使用程度，必须为[0, 1]区间内的浮点数，0代表不使用优先级。（``alpha`` ``beta`` ``anneal_step``三个参数均请参考论文*Prioritized Experience Replay*） |
| replay_buffer.beta                   | float          | replay buffer中优先级修正的使用程度，必须为[0, 1]区间内的浮点数，0代表不使用修正。 |
| replay_buffer.anneal_step            | float          | replay buffer中beta需要多少步从初始值退火到1，``float("inf")``代表不退火。 |
| replay_buffer.deepcopy               | bool           | replay buffer中，sample出的数据是否采用深拷贝，以防止buffer外的操作改变buffer内的数据。 |

**实例**

```python
pong_dqn_default_config = dict(
	# ...
	replay_buffer=dict(
        buffer_name=['agent'],
        agent=dict(
            meta_maxlen=10000,
            max_reuse=100,
            min_sample_ratio=1,
        ),
    ),
    # ...
)
```



## actor

| Name                     | Type  | Description                                                  |
| ------------------------ | ----- | ------------------------------------------------------------ |
| actor.n_sample           | int   | ``n_sample``与``n_episode``，任选一项填写即可。actor对轨迹数据进行处理后，得到的一条可用于训练的数据称为称为一个sample。``n_sample``表示当actor采集到这么多sample后就会暂停，转到learner训练。 |
| actor.n_episode          | int   | ``n_sample``与``n_episode``，任选一项填写即可。``n_episode``表示当actor与环境互动这么多episode后就会暂停，转到learner训练。 |
|                          |       | actor在collect任务中，会根据``traj_len``与环境交互得到轨迹，处理后得到一些训练数据，但这些数据不一定能满足``n_sample``或是``n_episode``，故actor可能会与环境按``traj_len``交互多次，直到满足``n_sample``或是``n_episode``。无论是``n_sample``还是``n_episode``，再从learner转回actor工作时，环境都会继续之前的状态运行，而非重启环境。 |
| actor.traj_len           | float | 请参考``policy.collect.traj_len``。二者的值需要保持相同。    |
| actor.collect_print_freq | int   | actor端按照``n_sample``或``n_episode``得到一定的数据成为一个collect step。该项便是控制每隔多少个collect step打印一次collect相关的信息。 |

**实例**

```python
pong_dqn_default_config = dict(
	# ...
	actor=dict(
        n_sample=100,
        traj_len=traj_len,
        traj_print_freq=100,
        collect_print_freq=100,
    ),
    # ...
)
```



## evaluator

| Name                | Type  | Description                                                  |
| ------------------- | ----- | ------------------------------------------------------------ |
| evaluator.n_episode | int   | evaluator对policy进行评估时，所有环境加在一起，总共运行多少个episode。 |
| evaluator.eval_freq | int   | serial entry中，每隔多少个learner的train iteration，调用evaluator进行一次policy评估。 |
| evaluator.stop_val  | float | 当某一次评估的平均reward超过该值后，认为算法已经达到目标效果，结束训练过程。 |

**实例**

```python
pong_dqn_default_config = dict(
	# ...
	evaluator=dict(
        n_episode=4,
        eval_freq=5000,
        stop_val=20,
    ),
    # ...
)
```



## learner

| Name                  | Type           | Description                                                  |
| --------------------- | -------------- | ------------------------------------------------------------ |
| learner.load_path     | str            | learner端从何路径载入某个checkpoint的网络参数，``''``表示不载入checkpoint。 |
| learner.hook          | Dict[str, Any] | learner hook的配置项。nerveX中已经实现的hook包括：``LoadCkptHook`` ``SaveCkptHook`` ``LogShowHook`` ``LogReduceHook``等。hook会在iteration的前后，或是整个训练的前后被调用，以完成特定任务。 |
| learner.hook.log_show | Dict[str, Any] | 下面以一个hook为例，介绍如何填写配置项。首先，key仅代表该hook的名字，需与value中的``name``保持一致。``type``表示该hook的类型，需为hook的注册名。``position``表示该hook的调用位置。``priority``决定该hook在同调用位置的hook中的调用时机。``ext_args``包含其他的信息，请参考对应hook的源代码。 |

**实例**

```python
pong_dqn_default_config = dict(
	# ...
	learner=dict(
        load_path='',
        hook=dict(
            log_show=dict(
                name='log_show',
                type='log_show',
                priority=20,
                position='after_iter',
                ext_args=dict(freq=100, ),
            ),
        ),
    ),
    # ...
)
```



## commander

暂无配置项。

