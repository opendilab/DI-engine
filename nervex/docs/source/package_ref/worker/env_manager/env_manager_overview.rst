Env Manager Overview
========================


Env Manager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

概述：
    env manager是一个向量化的环境管理器，其中同时运行多个相同类型不同配置的环境，实际实现方式包含子进程向量化和伪向量化（循环串行）两种模式
代码结构：
    主要分为如下几个子模块：

        1. base_env_manager: base_env_manager可以同时维护、使用多个环境实例，提供与env相似的接口，通过伪向量化（循环串行）的方式对环境进行运行。
        2. vec_env_manager: vec_env_manager继承了base_env_manager的接口，通过子进程向量化的方式，即调用multiprocessing，通过子进程进程间通信的方式对环境进行管理和运行。

基类定义：
    1. BaseEnvManager (nervex/worker/actor/env_manager/base_env_manager.py)

        .. code:: python

            class BaseEnvManager(ABC):
                def __init__(self, env_fn: Callable, env_cfg: Iterable, env_num: int) -> None:
                    self._env_num = env_num
                    self._envs = [env_fn(c) for c in env_cfg]
                    assert len(self._envs) == self._env_num

                def __getattr__(self, key: str) -> Any:
                    """
                    Note: if a python object doesn't have the attribute named key, it will call this method
                    """
                    return [getattr(env, key) if hasattr(env, key) else None for env in self._envs]

                @property
                def env_num(self) -> int:
                    return self._env_num

                def reset(self,
                        reset_param: Union[None, List[dict]] = None,
                        env_id: Union[None, List[int]] = None) -> Union[list, dict]:
                    self._env_done = {}
                    for i in (env_id if env_id is not None else range(self.env_num)):
                        self._env_done[i] = False
                    obs = self._execute_by_envid('reset', param=reset_param, env_id=env_id)
                    return self._envs[0].pack(obs=obs)

                def step(self, action: List[Any], env_id: Union[None, List[int]] = None) -> Union[list, dict]:
                    param = self._envs[0].unpack(action)
                    ret = self._execute_by_envid('step', param=param, env_id=env_id)
                    if isinstance(ret, list):
                        for i, t in enumerate(ret):
                            self._env_done[i] = t.done
                    elif isinstance(ret, dict):
                        for k, v in ret.items():
                            self._env_done[k] = v.done
                    return self._envs[0].pack(timesteps=ret)

                def seed(self, seed: List[int], env_id: Union[None, List[int]] = None) -> None:
                    param = [{'seed': s} for s in seed]
                    return self._execute_by_envid('seed', param=param, env_id=env_id)

                def _execute_by_envid(
                        self,
                        fn_name: str,
                        param: Union[None, List[dict]] = None,
                        env_id: Union[None, List[int]] = None
                ) -> Union[list, dict]:
                    real_env_id = list(range(self.env_num)) if env_id is None else env_id
                    if param is None:
                        ret = {real_env_id[i]: getattr(self._envs[real_env_id[i]], fn_name)() for i in range(len(real_env_id))}
                    else:
                        ret = {
                            real_env_id[i]: getattr(self._envs[real_env_id[i]], fn_name)(**param[i])
                            for i in range(len(real_env_id))
                        }
                    ret = list(ret.values()) if env_id is None else ret
                    return ret

                def close(self) -> None:
                    for env in self._envs:
                        env.close()

                @property
                def all_done(self) -> bool:
                    return all(self._env_done.values())

                @property
                def env_done(self) -> List[bool]:
                    return self._env_done


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
                def __init__(self, *args, **kwargs) -> None:
                    super(SubprocessEnvManager, self).__init__(*args, **kwargs)
                    self._parent_remote, self._child_remote = zip(*[Pipe() for _ in range(self.env_num)])
                    self._processes = [
                        Process(target=self.worker_fn, args=(parent, child, CloudpickleWrapper(env)), daemon=True)
                        for parent, child, env in zip(self._parent_remote, self._child_remote, self._envs)
                    ]
                    for p in self._processes:
                        p.start()
                    for c in self._child_remote:
                        c.close()
                    self._closed = False

                @staticmethod
                def worker_fn(p, c, env_wrapper) -> None:
                    env = env_wrapper.data
                    p.close()
                    try:
                        while True:
                            cmd, data = c.recv()
                            if cmd == 'getattr':
                                c.send(getattr(env, data) if hasattr(env, data) else None)
                            elif cmd in ['reset', 'step', 'seed', 'close']:
                                if data is None:
                                    c.send(getattr(env, cmd)())
                                else:
                                    c.send(getattr(env, cmd)(**data))
                                if cmd == 'close':
                                    c.close()
                                    break
                            else:
                                c.close()
                                raise KeyError("not support env cmd: {}".format(cmd))
                    except KeyboardInterrupt:
                        c.close()

                # override
                def _execute_by_envid(
                        self,
                        fn_name: str,
                        param: Union[None, List[dict]] = None,
                        env_id: Union[None, List[int]] = None
                ) -> Union[list, dict]:
                    real_env_id = list(range(self.env_num)) if env_id is None else env_id
                    for i in range(len(real_env_id)):
                        if param is None:
                            self._parent_remote[real_env_id[i]].send([fn_name, None])
                        else:
                            self._parent_remote[real_env_id[i]].send([fn_name, param[i]])
                    ret = {i: self._parent_remote[i].recv() for i in real_env_id}
                    ret = list(ret.values()) if env_id is None else ret
                    return ret

                # override
                def __getattr__(self, key: str) -> Any:
                    for p in self._parent_remote:
                        p.send(['getattr', key])
                    return [p.recv() for p in self._parent_remote]

                # override
                def close(self) -> None:
                    if self._closed:
                        return
                    super().close()
                    for p in self._parent_remote:
                        p.send(['close', None])
                    result = [p.recv() for p in self._parent_remote]
                    for p in self._processes:
                        p.join()
                    self._closed = True


        - 概述：

            继承了BaseEnvManager，将单机上的循环执行环境改为了通过调用multiprocessing的进行子进程通信，因此SubprocessEnvManager能一定程度上提升环境的产出速度。

        - 类接口方法：
           使用时，同BaseEnvManager基本相同，即使用时可以参考以cartpole为例的如下代码:

        .. code:: python
            def _setup_env(self):
                env_num = self.cfg.env.env_num
                self.env = SubprocessEnvManager(CartpoleEnv, env_cfg=[self.cfg.env for _ in range(env_num)], env_num=env_num)


.. note::
    BaseEnvManager和SubprocessEnvManager相关插件的测试可以参见 `nervex/worker/actor/env_manager/tests/test_base_env_manager.py` 和 `nervex/worker/actor/env_manager/tests/test_vec_env_manager.py`。
