Learner Overview
===================


Learner 
^^^^^^^^^^

概述：
    Learner是RL算法进行训练的核心。在串行与并行两个版本中，learner有着不同的调用接口和运行模式。

代码结构：
    主要分为如下几个子模块：

        1. learner: 可以接收训练数据，进行训练（包括模型更新、策略更新、经验回放池更新等）。
        2. hook: 用于在训练的特定时间点执行一些函数，如加载模型、保存模型、打印log等
        3. comm: 用于并行训练中和 coordinator 通信。

基类定义：
    1. BaseLearner (worker/learner/base_learner.py)

        .. code:: python

            class BaseLearner(object):
                r"""
                Overview:
                    Base class for model learning.
                Interface:
                    __init__, register_hook, train, start, setup_dataloader, close, call_hook, save_checkpoint
                Property:
                    learn_info, priority_info, last_iter, name, rank, policy
                    tick_time, monitor, log_buffer, logger, tb_logger, load_path
                """

                _name = "BaseLearner"  # override this variable for sub-class learner

                def __init__(self, cfg: EasyDict) -> None:
                    self._cfg = deep_merge_dicts(base_learner_default_config, cfg)
                    self._learner_uid = get_task_uid()
                    self._load_path = self._cfg.load_path
                    self._use_cuda = self._cfg.get('use_cuda', False)
                    self._use_distributed = self._cfg.use_distributed

                    # Learner rank. Used when there are more than one learner.
                    self._rank = get_rank()
                    self._device = 'cuda:{}'.format(self._rank % 8) if self._use_cuda else 'cpu'

                    # Logger (Monitor is initialized with policy setter)
                    # Only rank == 0 learner needs monitor and tb_logger, else only needs text_logger to display terminal output.
                    self._timer = EasyTimer()
                    rank0 = True if self._rank == 0 else False
                    self._logger, self._tb_logger = build_logger('./log/learner', 'learner', rank0)
                    self._log_buffer = build_log_buffer()
                
                    # Checkpoint helper. Used to save model checkpoint.
                    self._checkpointer_manager = build_checkpoint_helper(self._cfg)
                    # Learner hook. Used to do specific things at specific time point. Will be set in ``_setup_hook``
                    self._hooks = {'before_run': [], 'before_iter': [], 'after_iter': [], 'after_run': []}
                    # Priority info. Used to update replay buffer according to data's priority.
                    self._priority_info = None
                    # Last iteration. Used to record current iter.
                    self._last_iter = CountVar(init_val=0)
                    self.info(pretty_print({
                        "config": self._cfg,
                    }, direct_print=False))

                    # Setup wrapper and hook.
                    self._setup_wrapper()
                    self._setup_hook()

                def _setup_hook(self) -> None:
                    if hasattr(self, '_hooks'):
                        self._hooks = merge_hooks(self._hooks, build_learner_hook_by_cfg(self._cfg.hook))
                    else:
                        self._hooks = build_learner_hook_by_cfg(self._cfg.hook)

                def _setup_wrapper(self) -> None:
                    self._wrapper_timer = EasyTimer()
                    self.train = self.time_wrapper(self.train, 'train_time')

                def time_wrapper(self, fn: Callable, name: str):

                    def wrapper(*args, **kwargs) -> Any:
                        with self._wrapper_timer:
                            ret = fn(*args, **kwargs)
                        self._log_buffer[name] = self._wrapper_timer.value
                        return ret

                    return wrapper

                def register_hook(self, hook: LearnerHook) -> None:
                    add_learner_hook(self._hooks, hook)

                def train(self, data: dict) -> None:
                    assert hasattr(self, '_policy'), "please set learner policy"
                    self.call_hook('before_iter')
                    # Pre-process data
                    with self._timer:
                        data = self._policy.data_preprocess(data)
                    # Forward
                    log_vars = self._policy.forward(data)
                    # Update replay buffer's priority info
                    priority = log_vars.pop('priority', None)
                    replay_buffer_idx = [d.get('replay_buffer_idx', None) for d in data]
                    replay_unique_id = [d.get('replay_unique_id', None) for d in data]
                    self._priority_info = {
                        'replay_buffer_idx': replay_buffer_idx,
                        'replay_unique_id': replay_unique_id,
                        'priority': priority
                    }
                    # Update log_buffer
                    log_vars['data_preprocess_time'] = self._timer.value
                    self._log_buffer.update(log_vars)
                    
                    self.call_hook('after_iter')
                    self._last_iter.add(1)

                @auto_checkpoint
                def start(self) -> None:
                    self._finished_task = None
                    # before run hook
                    self.call_hook('before_run')

                    max_iterations = self._cfg.max_iterations
                    for _ in range(max_iterations):
                        data = self._next_data()
                        self.train(data)

                    self._finished_task = {'finish': True}
                    # after run hook
                    self.call_hook('after_run')

                def setup_dataloader(self) -> None:
                    cfg = self._cfg.dataloader
                    self._dataloader = AsyncDataLoader(
                        self.get_data,
                        cfg.batch_size,
                        self._device,
                        cfg.chunk_size,
                        collate_fn=lambda x: x,
                        num_workers=cfg.num_workers
                    )
                    self._next_data = self.time_wrapper(self._next_data, 'data_time')

                def _next_data(self) -> Any:
                    return next(self._dataloader)

                def close(self) -> None:
                    if hasattr(self, '_dataloader'):
                        del self._dataloader
                    self._tb_logger.close()

                def call_hook(self, name: str) -> None:
                    for hook in self._hooks[name]:
                        hook(self)

                def save_checkpoint(self) -> None:
                    names = [h.name for h in self._hooks['after_run']]
                    assert 'save_ckpt_after_run' in names
                    idx = names.index('save_ckpt_after_run')
                    self._hooks['after_run'][idx](self)


        - 概述：
            learner基类，是串行模式与并行模式中进行训练的核心。

        - 接口方法：
            1. __init__: 初始化
            2. train: 传入训练数据，训练一个迭代，可被串行pipeline或 ``start`` 调用。
            3. start: 训练多个迭代，每个迭代中自行发送获取数据的请求，拿到数据后调用 ``train`` 进行训练，可被并行pipeline调用。
            4. setup_dataloader: 为并行训练设置dataloader。
            5. close: 正确关闭各项资源。
            6. call_hook: 根据传入的hook位置名，调用该位置所有hook。
            7. register_hook: 注册新的hook。
            8. save_checkpoint: 调用hook保存checkpoint。

            .. note::

                在 **串行pipeline** 中，learner与collector交替工作（同步），故 ``train`` 方法是从外界传入训练数据，由learner训练一个迭代。
                
                而在 **并行pipeline** 中，learner与collector同一时刻都在工作（异步），故 ``start`` 方法可作为一个线程启动，自行从dataloader获取数据（所以dataloader是并行pipeline特有的，串行没有），根据预先设定的最大迭代数及evaluate收敛情况，训练多个迭代。其中每一个迭代在获取数据后，都调用 ``train`` 进行当前迭代的训练。


    2. Hook 与 LearnerHook (worker/learner/learner_hook.py)

        .. code:: python

            class Hook(ABC):

                def __init__(self, name: str, priority: float, **kwargs) -> None:
                    self._name = name
                    assert priority >= 0, "invalid priority value: {}".format(priority)
                    self._priority = priority

                @property
                def name(self) -> str:
                    return self._name

                @property
                def priority(self) -> float:
                    return self._priority

                @abstractmethod
                def __call__(self, engine: Any) -> Any:
                    raise NotImplementedError


            class LearnerHook(Hook):
                positions = ['before_run', 'after_run', 'before_iter', 'after_iter']

                def __init__(self, *args, position: str, **kwargs) -> None:
                    super().__init__(*args, **kwargs)
                    assert position in self.positions
                    self._position = position

                @property
                def position(self) -> str:
                    return self._position



        - 概述：
            Hook是最基本的基类，仅定义名字name和优先度priority。
            LearnerHook是在其基础上针对learner的封装，考虑到learner可能需要在整个训练前后，及每一个迭代前后执行一些函数，而添加了位置position这一属性，该属性取值必须为类变量positions中的一个。

        - 类接口方法：
            1. __init__: 初始化。
            2. __call__: 调用hook要执行的函数。（子类必须重写实现该方法）

    3. BaseCommLearner (worker/learner/comm/base_comm_learner.py)

        .. code:: python

            class BaseCommLearner(ABC):

                def __init__(self, cfg: 'EasyDict') -> None:  # noqa
                    self._cfg = cfg
                    self._learner_uid = get_task_uid()
                    self._timer = EasyTimer()
                    if cfg.use_distributed:
                        self._rank, self._world_size = dist_init()
                    else:
                        self._rank, self._world_size = 0, 1
                    self._use_distributed = cfg.use_distributed
                    self._end_flag = True

                @abstractmethod
                def send_policy(self, state_dict: dict) -> None:
                    raise NotImplementedError

                @abstractmethod
                def get_data(self, batch_size: int) -> list:
                    raise NotImplementedError

                @abstractmethod
                def send_learn_info(self, learn_info: dict) -> None:
                    raise NotImplementedError

                def start(self) -> None:
                    self._end_flag = False

                def close(self) -> None:
                    self._end_flag = True
                    if self._use_distributed:
                        dist_finalize()

                @abstractproperty
                def hooks4call(self) -> list:
                    raise NotImplementedError

                def _create_learner(self, task_info: dict) -> BaseLearner:
                    # Prepare learner config and instantiate a learner object.
                    learner_cfg = EasyDict(task_info['learner_cfg'])
                    learner_cfg['use_distributed'] = self._use_distributed
                    learner = BaseLearner(learner_cfg)
                    # Set 3 methods and dataloader in created learner that are necessary in parallel setting.
                    for item in ['get_data', 'send_policy', 'send_learn_info']:
                        setattr(learner, item, getattr(self, item))
                    learner.setup_dataloader()
                    # Set policy in created learner.
                    policy_cfg = task_info['policy']
                    policy_cfg['use_distributed'] = self._use_distributed
                    learner.policy = create_policy(policy_cfg, enable_field=['learn']).learn_mode
                    return learner

        - 概述：
            base learner可以独立完成串行pipeline中的训练工作，但对于并行pipeline来说，虽然提供了训练接口，但还有一些问题尚未解决，如数据怎么获得，如何与外界通信等等，comm learner便是负责解决并行模式中的这些问题的。

            comm learner并不实际进行训练，其持有一个base learner，并为其解决涉及通信的问题，依然由base learner进行训练。

            .. note::

                故串行pipeline可以实例化base learner并直接对其操作；但在并行pipeline中应当实例化comm learner，再由comm learner通过 ``_create_learner`` 创建base learner。

            在并行训练模式中，learner需要自己发出数据请求、定时将当前策略及训练信息发送出去，这些操作将以hook的方式完成，而comm learner的一个重要工作就是将这些hook及执行hook时所需要的函数注册至learner中，即在 ``hooks4call`` 中返回上述hook，并实现 ``get_data`` , ``send_policy`` ,  ``send_learn_info`` 三个方法hook中需要用到的方法。
        
        - 类变量：
            无

        - 类接口方法：
            1. __init__：初始化
            2. start：开启comm learner服务
            3. close：关闭comm learner服务

        - 子类需继承重写方法：
            1. get_data: 获取数据的函数，AyncDataLoader的参数
            2. send_policy: 将策略存储或发送
            3. send_learn_info: 将训练信息存储或发送
            4. hooks4call: 策略与训练信息的定时存储或发送的hooks dict


并行模式中的训练流程解析：
    相对于简单直接的串行模式，并行模式由于涉及到异步运行的learner collector之间的通信问题，更加晦涩难懂。故在这一部分以我们实现的 **FlaskFileSystemLearner(worker/learner/comm/flask_fs_learner.py)** ——这一使用flask及文件系统进行通信的comm learner——为例，来介绍并行模式中从并行pipeline入口部署coordinator, comm learner开始，到二者建立通信连接，再到coordinator启动comm learner并为其一次或多次分配任务，到最终二者关闭通信连接的流程。

        .. image:: images/parallel_learner_sequence.jpg

        上图即展示了coordinator和comm learner从被并行pipeline部署，到建立连接，到实际任务分配与执行，再到最后断开连接的过程。至于实际任务的分配与执行，请继续阅读。

    在介绍FlaskFileSystemLearner前，还有必要介绍一下LearnerSlave，这一真正负责和coordinator进行通信的类。LearnerSlave继承自Slave，其master为coordinator中的变量master，负责和coordinator通信，处理master发来的task，并利用FlaskFileSystemLearner传来的回调函数响应相应的task。其本质是利用master-slave机制帮助FlaskFileSystemLearner完成与coordinator的通信工作。

    BaseCommLearner, FlaskFileSystemLearner, BaseLearner, LearnerSlave这几个类之间的关系可见类图所示(本类图并不完整，仅包含为理解后述工作流程所必须的部分)：

        .. image:: images/comm_learner_class.jpg

    然后我们开始介绍并行模式下的FlaskFileSystemLearner这一comm learner的工作流程，即实际任务的分配与执行过程，也即第一张顺序图中被略去的部分。可以参考以下顺序图帮助理解。

        .. image:: images/comm_learner_sequence.jpg


    1. comm learner的创建
        并行pipeline会创建comm learner，并调用 ``start`` 方法以启动comm learner服务
        
        comm learner中先是实例化一个 **learner slave** ，将自己的四个函数作为回调函数传给learner slave（至于什么是回调函数及回调函数是用来做什么的，我们在后边的流程中再解释），learner slave会通过预先商定的ip地址与端口号与coordinator建立连接。
        
        此外，comm learner创建几个 **长度为1的队列** ，用于存放一些和通信相关的消息字典。

    2. learner的创建
        在coordinator发来任务之前，comm learner及learner slave一直都处于待命状态。一旦coordinator发来任务，learner slave的 ``_process_task`` 就会接收到该任务。
        
        coordinator知道comm learner的工作流程为： **首先建立learner，然后重复执行获取数据、利用数据训练这一过程，直到训练结束** 。故此时的任务应当为 ``learner_start_task`` ，此外还传来建立learner必须的信息。
        
        这些信息都传到了learner slave处，但learner的创建是在comm learner中完成的，这就用到了我们刚刚提到的 **回调函数** 。回调函数由comm learner实现，但作为参数传递给learner slave，故learner slave可以调用这些函数。
        
        对于 ``learner_start_task`` ，learner slave调用comm learner的 ``deal_with_learner_start`` 方法，完成建立learner的工作。完成后，learner slave向coordinator返回成功的信息。

    3. learner get data
        learner在建立后，dataloader便会调用comm learner中实现的 ``get_data`` 方法 **试图获取数据** ， ``get_data`` 中会在comm learner的 ``_data_demand_queue`` 放入这一数据请求，然后试图从 ``_data_result_queue`` 中取出数据，若其为空，就被 **阻塞** 在了这里。
        
        视线回到coordinator，当coordinator收到流程2中最后 ``learner_start_task`` 成功执行的信息后，发送任务 ``learner_get_data_task`` ，learner slave调用comm learner中的 ``deal_with_get_data`` ，从 ``_data_demand_queue`` 中取出数据请求，并返回给coordinator。

    4. learner learn
        coordinator在收到learner的数据请求后，会发送 ``learner_learn_task`` 给learner slave，其中就包含了learner请求的 **数据** （或元数据） 。learner slave收到后调用comm learner的 ``deal_with_learner_learn`` 方法，将收到的数据信息放入 ``_data_result_queue`` 中，并等待learner结束训练，可以从 ``_learn_info_queue`` 中获取训练信息。

        视线回到learner，learner是因为dataloader无法获得数据而被阻塞住的，现在 ``_data_result_queue`` 中有了数据信息，dataloader可以将其取出，处理成learner需要的格式，交由learner **训练一个迭代** 。训练完成后，learner将训练信息存放在 ``_learn_info_queue`` 当中。

        视线回到comm learner的 ``deal_with_learner_learn`` 方法，它从 ``_learn_info_queue`` 取出训练信息，并将其通过learner slave返回给coordinator。对于该信息的内容有 **两种情况** ：

            - learner没有完成训练，需要继续迭代：此时dataloader又会调用 ``get_data`` ，coordinator也会在收到该信息后继续发送任务 ``learner_get_data_task`` ，便回到了流程3。

            - learner完成训练：comm learner中会将learner关闭，等待coordinator再次分配新的任务 ``learner_start_task`` ，完成新的训练工作，便回到了流程2。

    5. comm learner close
        可以通过输入命令的方式手动关闭comm learner；否则comm learner将 **常驻** ，等待coordinator分配新的任务，执行后返回结果。
