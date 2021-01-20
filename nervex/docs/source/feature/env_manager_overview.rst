Env Manager Overview
========================


Env Manager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

概述：
    env manager是一个环境管理器，可以管理多个相同类型不同配置的环境。目前支持的类型有单进程串行和多进程并行两种模式。
代码结构：
    主要分为如下几个子模块：

        1. base_env_manager: base_env_manager通过循环串行（伪并行）来维护多个环境实例，提供与env相似的接口。
        2. vec_env_manager: 继承base_env_manager的接口，通过子进程向量化的方式，即调用multiprocessing，通过子进程进程间通信的方式对环境进行管理和运行。

基类定义：
    1. BaseEnvManager (nervex/worker/actor/env_manager/base_env_manager.py)

        .. code:: python

            class BaseEnvManager(ABC):

                def __init__(
                        self,
                        env_fn: Callable,
                        env_cfg: Iterable,
                        env_num: int,
                        episode_num: Optional[int] = 'inf',
                ) -> None:
                    self._env_num = env_num
                    self._env_fn = env_fn
                    self._env_cfg = env_cfg
                    if episode_num == 'inf':
                        episode_num = float('inf')
                    self._epsiode_num = episode_num
                    self._transform = partial(to_ndarray)
                    self._inv_transform = partial(to_tensor, dtype=torch.float32)
                    self._closed = True
                    # env_ref is used to acquire some common attributes of env, like obs_shape and act_shape
                    self._env_ref = self._env_fn(self._env_cfg[0])

                def _create_state(self) -> None:
                    self._closed = False
                    self._env_episode_count = {i: 0 for i in range(self.env_num)}
                    self._env_done = {i: False for i in range(self.env_num)}
                    self._next_obs = {i: None for i in range(self.env_num)}
                    self._envs = [self._env_fn(c) for c in self._env_cfg]
                    assert len(self._envs) == self._env_num

                def _check_closed(self):
                    assert not self._closed, "env manager is closed, please use the alive env manager"

                @property
                def env_num(self) -> int:
                    return self._env_num

                @property
                def next_obs(self) -> Dict[int, Any]:
                    return self._inv_transform({i: self._next_obs[i] for i, d in self._env_done.items() if not d})

                @property
                def done(self) -> bool:
                    return all([v == self._epsiode_num for v in self._env_episode_count.values()])

                @property
                def method_name_list(self) -> list:
                    return ['reset', 'step', 'seed', 'close']

                def launch(self, reset_param: Union[None, List[dict]] = None) -> None:
                    assert self._closed, "please first close the env manager"
                    self._create_state()
                    self.reset(reset_param)

                def reset(self, reset_param: Union[None, List[dict]] = None) -> None:
                    if reset_param is None:
                        reset_param = [{} for _ in range(self.env_num)]
                    self._reset_param = reset_param
                    # set seed
                    if hasattr(self, '_env_seed'):
                        for env, s in zip(self._envs, self._env_seed):
                            env.seed(s)
                    for i in range(self.env_num):
                        self._reset(i)

                def _reset(self, env_id: int) -> None:
                    obs = self._safe_run(lambda: self._envs[env_id].reset(**self._reset_param[env_id]))
                    self._next_obs[env_id] = obs

                def _safe_run(self, fn: Callable):
                    try:
                        return fn()
                    except Exception as e:
                        self.close()
                        raise e

                def step(self, action: Dict[int, Any]) -> Dict[int, namedtuple]:
                    self._check_closed()
                    timesteps = {}
                    for env_id, act in action.items():
                        act = self._transform(act)
                        timesteps[env_id] = self._safe_run(lambda: self._envs[env_id].step(act))
                        if timesteps[env_id].done:
                            self._env_done[env_id] = True
                            self._env_episode_count[env_id] += 1
                        self._next_obs[env_id] = timesteps[env_id].obs
                    if not self.done and all([d for d in self._env_done.values()]):
                        for i in range(self.env_num):
                            self._reset(i)
                            self._env_done[i] = False
                    return self._inv_transform(timesteps)

                def seed(self, seed: List[int]) -> None:
                    if isinstance(seed, numbers.Integral):
                        seed = [seed + i for i in range(self.env_num)]
                    self._env_seed = seed

                def close(self) -> None:
                    if self._closed:
                        return
                    self._env_ref.close()
                    for env in self._envs:
                        env.close()
                    self._closed = True

        - 概述：

            使用循环串行的方式运行多个环境，通过调用env的对应接口（详见env overview）。

        - 类接口方法：
            1. __init__: 初始化
            2. reset: 不传入参数时默认reset所有环境，也可以传入list结构的env_id和reset子类的实现中的输入参数(e.g.比如一个episode结束重启时需要外部指定一些参数),对manager持有的某几个环境进行reset
            3. close: 关闭环境，释放资源，close所有环境
            4. step: 环境执行输入的动作，完成一个时间步，同reset一样，可以传入list结构的env_id对manager持有的某几个环境进行操作
            5. seed: 设置环境随机种子，可以传入list结构的env_id对manager持有的某几个环境设置特定的seed
            6. env_done: 哪几个持有的环境已经done即运行结束
            7. all_done: 是否所有持有的环境已经运行结束

        .. note::

            具体的使用可以参考测试文件 nervex/worker/actor/env_manager/tests/test_base_env_manager.py, 或者直接参考SubprocessEnvManager的使用方式（两者使用相同的接口）

    2. SubprocessEnvManager (nervex/worker/actor/env_manager/vec_env_manager.py)

        .. code:: python

            class SubprocessEnvManager(BaseEnvManager):

                def __init__(
                        self,
                        env_fn: Callable,
                        env_cfg: Iterable,
                        env_num: int,
                        episode_num: Optional[int] = 'inf',
                        timeout : Optional[float] = 0.01,
                        wait_num: Optional[int] = 2,
                ) -> None:
                    super().__init__(env_fn, env_cfg, env_num, episode_num)
                    self.shared_memory = self._env_cfg[0].get("shared_memory", True)
                    self.timeout = timeout
                    self.wait_num = wait_num

                def _create_state(self) -> None:
                    r"""
                    Overview:
                        Fork/spawn sub-processes and create pipes to convey the data.
                    """
                    self._closed = False
                    self._env_episode_count = {env_id: 0 for env_id in range(self.env_num)}
                    self._env_done = {env_id: False for env_id in range(self.env_num)}
                    self._next_obs = {env_id: None for env_id in range(self.env_num)}
                    if self.shared_memory:
                        obs_space = self._env_ref.info().obs_space
                        shape = obs_space.shape
                        dtype = np.dtype(obs_space.value['dtype']) if obs_space.value is not None else np.dtype(np.float32)
                        self._obs_buffers = {env_id: ShmBuffer(dtype, shape) for env_id in range(self.env_num)}
                    else:
                        self._obs_buffers = {env_id: None for env_id in range(self.env_num)}
                    self._parent_remote, self._child_remote = zip(*[Pipe() for _ in range(self.env_num)])
                    context_str = 'spawn' if platform.system().lower() == 'windows' else 'fork'
                    ctx = get_context(context_str)
                    # due to the runtime delay of lambda expression, we use partial for the generation of different envs,
                    # otherwise, it will only use the last item cfg.
                    env_fn = [partial(self._env_fn, cfg=self._env_cfg[env_id]) for env_id in range(self.env_num)]
                    self._processes = [
                        ctx.Process(
                            target=self.worker_fn,
                            args=(parent, child, CloudpickleWrapper(fn), obs_buffer, self.method_name_list),
                            daemon=True
                        ) for parent, child, fn, obs_buffer in
                        zip(self._parent_remote, self._child_remote, env_fn, self._obs_buffers.values())
                    ]
                    for p in self._processes:
                        p.start()
                    for c in self._child_remote:
                        c.close()
                    self._env_state = {env_id: EnvState.INIT for env_id in range(self.env_num)}
                    self._waiting_env = {'step': set()}
                    self._setup_async_args()

                def _setup_async_args(self) -> None:
                    r"""
                    Overview:
                        set up the async arguments utilized in the step().
                        wait_num: for each time the minimum number of env return to gather
                        timeout: for each time the minimum number of env return to gather
                    """
                    self._async_args = {
                        'step': {
                            'wait_num': self.wait_num,
                            'timeout': self.timeout
                        },
                    }

                @property
                def active_env(self) -> List[int]:
                    return [i for i, s in self._env_state.items() if s == EnvState.RUN]

                @property
                def ready_env(self) -> List[int]:
                    return [i for i in self.active_env if i not in self._waiting_env['step']]

                @property
                def next_obs(self) -> Dict[int, Any]:
                    no_done_env_idx = [i for i, s in self._env_state.items() if s != EnvState.DONE]
                    sleep_count = 0
                    while all([self._env_state[i] == EnvState.RESET for i in no_done_env_idx]):
                        print('VEC_ENV_MANAGER: all the not done envs are resetting, sleep {} times'.format(sleep_count))
                        time.sleep(1)
                        sleep_count += 1
                    return self._inv_transform({i: self._next_obs[i] for i in self.ready_env})

                @property
                def done(self) -> bool:
                    return all([s == EnvState.DONE for s in self._env_state.values()])

                def launch(self, reset_param: Union[None, List[dict]] = None) -> None:
                    assert self._closed, "please first close the env manager"
                    self._create_state()
                    self.reset(reset_param)

                def reset(self, reset_param: Union[None, List[dict]] = None) -> None:
                    if reset_param is None:
                        reset_param = [{} for _ in range(self.env_num)]
                    self._reset_param = reset_param
                    # set seed
                    if hasattr(self, '_env_seed'):
                        for i in range(self.env_num):
                            self._parent_remote[i].send(CloudpickleWrapper(['seed', [self._env_seed[i]], {}]))
                        ret = [p.recv().data for p in self._parent_remote]
                        self._check_data(ret)

                    # reset env
                    lock = threading.Lock()
                    reset_thread_list = []
                    for env_id in range(self.env_num):
                        reset_thread = PropagatingThread(target=self._reset, args=(env_id, lock))
                        reset_thread.daemon = True
                        reset_thread_list.append(reset_thread)
                    for t in reset_thread_list:
                        t.start()
                    for t in reset_thread_list:
                        t.join()

                def _reset(self, env_id: int, lock: Any) -> None:

                    @retry_wrapper
                    def reset_fn():
                        self._parent_remote[env_id].send(CloudpickleWrapper(['reset', [], self._reset_param[env_id]]))
                        obs = self._parent_remote[env_id].recv().data
                        self._check_data([obs], close=False)
                        if self.shared_memory:
                            obs = self._obs_buffers[env_id].get()
                        with lock:
                            self._env_state[env_id] = EnvState.RUN
                            self._next_obs[env_id] = obs

                    try:
                        reset_fn()
                    except Exception as e:
                        if self._closed:  # exception cased by main thread closing parent_remote
                            return
                        else:
                            self.close()
                            raise e

                def step(self, action: Dict[int, Any]) -> Dict[int, namedtuple]:
                    self._check_closed()
                    env_ids = list(action.keys())
                    assert all([self._env_state[env_id] == EnvState.RUN for env_id in env_ids]
                            ), 'current env state are: {}, please check whether the requested env is in reset or done'.format(
                                {env_id: self._env_state[env_id]
                                    for env_id in env_ids}
                            )

                    for env_id, act in action.items():
                        act = self._transform(act)
                        self._parent_remote[env_id].send(CloudpickleWrapper(['step', [act], {}]))

                    handle = self._async_args['step']
                    wait_num, timeout = min(handle['wait_num'], len(env_ids)), handle['timeout']
                    rest_env_ids = list(set(env_ids).union(self._waiting_env['step']))

                    ready_env_ids = []
                    ret = {}
                    cur_rest_env_ids = copy.deepcopy(rest_env_ids)
                    while True:
                        rest_conn = [self._parent_remote[env_id] for env_id in cur_rest_env_ids]
                        ready_conn, ready_ids = SubprocessEnvManager.wait(rest_conn, min(wait_num, len(rest_conn)), timeout)
                        cur_ready_env_ids = [cur_rest_env_ids[env_id] for env_id in ready_ids]
                        assert len(cur_ready_env_ids) == len(ready_conn)
                        ret.update({env_id: p.recv().data for env_id, p in zip(cur_ready_env_ids, ready_conn)})
                        self._check_data(ret.values())
                        ready_env_ids += cur_ready_env_ids
                        cur_rest_env_ids = list(set(cur_rest_env_ids).difference(set(cur_ready_env_ids)))
                        # at least one not done timestep or all the connection is ready
                        if any([not t.done for t in ret.values()]) or len(ready_conn) == len(rest_conn):
                            break

                    self._waiting_env['step']: set
                    for env_id in rest_env_ids:
                        if env_id in ready_env_ids:
                            if env_id in self._waiting_env['step']:
                                self._waiting_env['step'].remove(env_id)
                        else:
                            self._waiting_env['step'].add(env_id)

                    lock = threading.Lock()
                    for env_id, timestep in ret.items():
                        if self.shared_memory:
                            timestep = timestep._replace(obs=self._obs_buffers[env_id].get())
                        ret[env_id] = timestep
                        if timestep.done:
                            self._env_episode_count[env_id] += 1
                            if self._env_episode_count[env_id] >= self._epsiode_num:
                                self._env_state[env_id] = EnvState.DONE
                            else:
                                self._env_state[env_id] = EnvState.RESET
                                reset_thread = PropagatingThread(target=self._reset, args=(env_id, lock))
                                reset_thread.daemon = True
                                reset_thread.start()
                        else:
                            self._next_obs[env_id] = timestep.obs

                    return self._inv_transform(ret)

                # this method must be staticmethod, otherwise there will be some resource conflicts(e.g. port or file)
                # env must be created in worker, which is a trick of avoiding env pickle errors.
                @staticmethod
                def worker_fn(p, c, env_fn_wrapper, obs_buffer, method_name_list) -> None:
                    env_fn = env_fn_wrapper.data
                    env = env_fn()
                    p.close()
                    try:
                        while True:
                            try:
                                cmd, args, kwargs = c.recv().data
                            except EOFError:  # for the case when the pipe has been closed
                                c.close()
                                break
                            try:
                                if cmd == 'getattr':
                                    ret = getattr(env, args[0])
                                elif cmd in method_name_list:
                                    if cmd == 'step':
                                        timestep = env.step(*args, **kwargs)
                                        if obs_buffer is not None:
                                            assert isinstance(timestep.obs, np.ndarray), type(ret)
                                            obs_buffer.fill(timestep.obs)
                                            timestep = timestep._replace(obs=None)
                                        ret = timestep
                                    elif cmd == 'reset':
                                        ret = env.reset(*args, **kwargs)  # obs
                                        if obs_buffer is not None:
                                            assert isinstance(ret, np.ndarray), type(ret)
                                            obs_buffer.fill(ret)
                                            ret = None
                                    elif args is None and kwargs is None:
                                        ret = getattr(env, cmd)()
                                    else:
                                        ret = getattr(env, cmd)(*args, **kwargs)
                                else:
                                    raise KeyError("not support env cmd: {}".format(cmd))
                                c.send(CloudpickleWrapper(ret))
                            except Exception as e:
                                # when there are some errors in env, worker_fn will send the errors to env manager
                                # directly send error to another process will lose the stack trace, so we create a new Exception
                                c.send(
                                    CloudpickleWrapper(
                                        e.__class__(
                                            '\nEnv Process Exception:\n' + ''.join(traceback.format_tb(e.__traceback__)) + repr(e)
                                        )
                                    )
                                )
                            if cmd == 'close':
                                c.close()
                                break
                    except KeyboardInterrupt:
                        c.close()

                def _check_data(self, data: Iterable, close: bool = True) -> None:
                    for d in data:
                        if isinstance(d, Exception):
                            # when receiving env Exception, env manager will safely close and raise this Exception to caller
                            if close:
                                self.close()
                            raise d

                # override
                def close(self) -> None:
                    if self._closed:
                        return
                    self._closed = True
                    self._env_ref.close()
                    for p in self._parent_remote:
                        p.send(CloudpickleWrapper(['close', None, None]))
                    for p in self._processes:
                        p.join()
                    for p in self._processes:
                        p.terminate()
                    for p in self._parent_remote:
                        p.close()


        - 概述：

            继承了BaseEnvManager，通multiprocessing模块为每个环境创建单独的进程，能加速数据产出速度。

        - 类接口方法：
           使用时，同BaseEnvManager基本相同。此外，
            1. wait_num 指定每次产出数据至少包含的环境数量， timeout指定最少等待时间。用户可以根据环境运行速度的快慢来调整这些参数。
            2. shared_memory 可以加速传递环境返回的大向量，对于环境返回的obs等变量大小超过100kB的时候，推荐设置为True。
            3. worker_fn 作为子进程的执行函数，创建env，并接受来自父进程中env_manager的指令。
            4. wait 等待环境返回。
            5. 每次调用需先通过 next_obs 函数得到可获得的env id和obs，再调用step 函数传入env id对应的action

           使用时可以参考如下代码:

        .. code:: python
        
            def _setup_env(self):
                env_num = self.cfg.env.env_num
                self.env = SubprocessEnvManager(CartpoleEnv, env_cfg=[self.cfg.env for _ in range(env_num)], env_num=env_num)


.. note::
    BaseEnvManager和SubprocessEnvManager相关插件的测试可以参见 `nervex/worker/actor/env_manager/tests/test_base_env_manager.py` 和 `nervex/worker/actor/env_manager/tests/test_vec_env_manager.py`。
