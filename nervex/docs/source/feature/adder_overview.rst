Adder Overview
===================


Adder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

概述：
    Adder的概念可以参考大规模强化学习训练框架ACME中的定义，有关ACME的介绍可见 `RLwarmup部分 <../rl_warmup/algorithm/large-scale-rl.html>`_ 。
    
    总的来说，adder的作用是将数据从collector取出加入dataset(buffer)之前进行一些预处理和聚合，将数据聚合送入replay buffer中，并且对数据进行一定程度的reduction/transformation。

    通过adder，我们可以方便的实现不同种类的数据预处理和聚合。

代码结构：
    我们统一将从 ``collector`` 取出的数据在加入buffer前进行预处理和聚合的函数放在 ``Adder`` 类中，代码位于 `rl_utils/adder.py` 。

类定义：
    Adder (rl_utils/adder.py)

        .. code:: python

            class Adder(object):
                """
                Overview:
                    Adder is a component that handles different transformations and calculations for transitions
                    in Collector Module(data generation and processing), such as GAE, n-step return, transition sampling etc.
                Interface:
                    __init__, get_traj, get_gae, get_gae_with_default_last_value, get_nstep_return_data, get_train_sample
                """

                def __init__(
                        self,
                        use_cuda: bool,
                        unroll_len: int,
                        last_fn_type: str = 'last',
                        null_transition: Optional[dict] = None,
                ) -> None:
                    """
                    Overview:
                        Initialization method for an adder instance
                    Arguments:
                        - use_cuda (:obj:`bool`): whether use cuda in all the operations
                        - unroll_len (:obj:`int`): learn training unroll length
                        - last_fn_type (:obj:`str`): the method type name for dealing with last residual data in a traj \
                            after splitting, should be in ['last', 'drop', 'null_padding']
                        - null_transition (:obj:`Optional[dict]`): dict type null transition, used in ``null_padding``
                    """
                    # ...
                    

                def _get_null_transition(self, template: dict) -> dict:
                    """
                    Overview:
                        Get null transition for padding. If ``self._null_transition`` is None, return input ``template`` instead.
                    Arguments:
                        - template (:obj:`dict`): the template for null transition.
                    Returns:
                        - null_transition (:obj:`dict`): the deepcopied null transition.
                    """
                    # ...

                def get_traj(self, data: deque, traj_len: int, return_num: int = 0) -> List:
                    """
                    Overview:
                        Get part of original deque type ``data`` as traj data for further process and sampling.
                    Arguments:
                        - data (:obj:`deque`): deque type traj data, should be the cache of traj
                        - traj_len (:obj:`int`): expected length of the collected trajectory, 'inf' means collecting will not \
                            end until episode is done
                        - return_num (:obj:`int`): number of datas which will be appended back to ``data``, determined by ``nstep``
                    Returns:
                        - traj (:obj:`List`): front(left) part of ``data``
                    """
                    # ...

                def get_gae(self, data: List[Dict[str, Any]], last_value: torch.Tensor, gamma: float,
                            gae_lambda: float) -> List[Dict[str, Any]]:
                    """
                    Overview:
                        Get GAE advantage for stacked transitions(T timestep, 1 batch). Call ``gae`` for calculation.
                    Arguments:
                        - data (:obj:`list`): transitions list, each element is a transition dict with at least ['value', 'reward']
                        - last_value (:obj:`torch.Tensor`): the last value(i.e.: the T+1 timestep)
                        - gamma (:obj:`float`): the future discount factor
                        - gae_lambda (:obj:`float`): gae lambda parameter
                    Returns:
                        - data (:obj:`list`): transitions list like input one, but each element owns extra advantage key 'adv'
                    """
                    # ...

                def get_gae_with_default_last_value(self, data: List[Dict[str, Any]], done: bool, gamma: float,
                                                    gae_lambda: float) -> List[Dict[str, Any]]:
                    """
                    Overview:
                        Like ``get_gae`` above to get GAE advantage for stacked transitions. However, this function is designed in
                        case ``last_value`` is not passed. If transition is not done yet, it wouold assign last value in ``data``
                        as ``last_value``, discard the last element in ``data``(i.e. len(data) would decrease by 1), and then call
                        ``get_gae``. Otherwise it would make ``last_value`` equal to 0.
                    Arguments:
                        - data (:obj:`List[Dict[str, Any]]`): transitions list, each element is a transition dict with \
                            at least['value', 'reward']
                        - done (:obj:`bool`): whether the transition reaches the end of an episode(i.e. whether the env is done)
                        - gamma (:obj:`float`): the future discount factor
                        - gae_lambda (:obj:`float`): gae lambda parameter
                    Returns:
                        - data (:obj:`List[Dict[str, Any]]`): transitions list like input one, but each element owns \
                            extra advantage key 'adv'
                    """
                    # ...

                def get_nstep_return_data(self, data: List[Dict[str, Any]], nstep: int, traj_len: int) -> List[Dict[str, Any]]:
                    """
                    Overview:
                        Process raw traj data by updating keys ['next_obs', 'reward', 'done'] in data's dict element.
                    Arguments:
                        - data (:obj:`List[Dict[str, Any]]`): transitions list, each element is a transition dict
                        - nstep (:obj:`int`): number of steps. If equals to 1, return ``data`` directly; \
                            Otherwise update with nstep value
                        - traj_len (:obj:`int`): expected length of the collected trajectory, 'inf' means collecting will not \
                            end until episode is done
                    Returns:
                        - data (:obj:`List[Dict[str, Any]]`): transitions list like input one, but each element updated with \
                            nstep value
                    """
                    # ...

                def get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                    """
                    Overview:
                        Process raw traj data by updating keys ['next_obs', 'reward', 'done'] in data's dict element.
                        If ``self._unroll_len`` equals to 1, which means no process is needed, can directly return ``data``.
                        Otherwise, ``data`` will be splitted according to ``self._unroll_len``, process residual part according to
                        ``self._last_fn_type`` and call ``lists_to_dicts`` to form sampled training data.
                    Arguments:
                        - data (:obj:`List[Dict[str, Any]]`): transitions list, each element is a transition dict
                    Returns:
                        - data (:obj:`List[Dict[str, Any]]`): transitions list processed after unrolling
                    """
                    # ...

        - 概述：
            Adder类内含有有各种对trajectory进行预处理和聚合操作的函数，其具体调用通常在 ``Policy`` 类中的 ``collect_mode.get_train_sample`` 即 ``self._get_train_sample`` 方法中。 
            为在 ``Policy`` 类中调用 ``adder``，我们需要在 ``Policy`` 类中的 ``_init_collect`` 方法中实例话 ``Adder`` 类。具体使用方式可见下例:

                .. code:: python

                    def _init_collect(self) -> None:
                        r"""
                        Overview:
                            Collect mode init moethod. Called by ``self.__init__``.
                            Init traj and unroll length, adder, collect armor.
                        """
                        # ...
                        self._adder = Adder(self._use_cuda, self._unroll_len)
                        # ...
                    
                    #...

                    def _get_train_sample(self, traj_cache: deque) -> Union[None, List[Any]]:
                        r"""
                        Overview:
                            Get the trajectory and the n step return data, then sample from the n_step return data

                        Arguments:
                            - traj_cache (:obj:`deque`): The trajectory's cache

                        Returns:
                            - samples (:obj:`dict`): The training samples generated
                        """
                        data = self._adder.get_traj(traj_cache, self._traj_len, return_num=self._collect_burnin_step)
                        data = self._adder.get_nstep_return_data(data, self._collect_nstep, self._traj_len)
                        return self._adder.get_train_sample(data)


        - 类方法：
            1. __init__: 初始化。
            2. get_traj: 该方法从 ``BaseSerialCollector`` 中的 trajectory cache pool 获得部分trajectory。 
            3. get_gae: 该方法根据trajectory计算相应的GAE advantage。
            4. get_gae_with_default_last_value: 该方法同样是根据trajectory计算相应的GAE advantage，不过也适用于没有结束的trajectory。
            5. get_nstep_return_data: 该方法获得多步的trajectory数据。
            6. get_train_sample: 该方法将数据转化为添加到buffer时需要的格式。
            7. 如有需要，可自行在 ``Adder`` 类下实现新方法。


.. note::
    Adder相关的测试可以参见 `rl_utils/tests/test_adder.py`


已经实现的模块:
    1. ``get_traj`` : 该方法从 ``BaseSerialCollector`` 中的 trajectory cache pool 获得traj_len长度的trajectory, 并返回该部分的trajectory。具体实现代码如下:
        

        .. code:: python

            def get_traj(self, data: deque, traj_len: int, return_num: int = 0) -> List:
                """
                Overview:
                    Get part of original deque type ``data`` as traj data for further process and sampling.
                Arguments:
                    - data (:obj:`deque`): deque type traj data, should be the cache of traj
                    - traj_len (:obj:`int`): expected length of the collected trajectory, 'inf' means collecting will not \
                        end until episode is done
                    - return_num (:obj:`int`): number of datas which will be appended back to ``data``, determined by ``nstep``
                Returns:
                    - traj (:obj:`List`): front(left) part of ``data``
                """
                num = min(traj_len, len(data))
                traj = [data.popleft() for _ in range(num)]
                for i in range(min(return_num, len(data))):
                    data.appendleft(copy.deepcopy(traj[-(i + 1)]))
                return traj
        

    该方法在 ``BaseSerialCollector`` 中通过 ``Policy`` 类的 ``collect_mode.get_train_sample`` 调用，输入的数据即为 ``collector`` 中的 ``traj_cache``:

        .. code:: python

            train_sample = self._policy.get_train_sample(self._traj_cache[env_id]) # traj_cache is the input of the get_traj function
    
    2. ``get_train_sample`` : 该方法同样在 ``BaseSerialCollector`` 中通过 ``Policy`` 类的 ``collect_mode.get_train_sample`` 调用，接受一个 ``list`` 结构的trajectory输入，返回可以放入buffer的训练数据。具体实现代码如下:

        .. code:: python

            def get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """
                Overview:
                    Process raw traj data by updating keys ['next_obs', 'reward', 'done'] in data's dict element.
                    If ``self._unroll_len`` equals to 1, which means no process is needed, can directly return ``data``.
                    Otherwise, ``data`` will be splitted according to ``self._unroll_len``, process residual part according to
                    ``self._last_fn_type`` and call ``lists_to_dicts`` to form sampled training data.
                Arguments:
                    - data (:obj:`List[Dict[str, Any]]`): transitions list, each element is a transition dict
                Returns:
                    - data (:obj:`List[Dict[str, Any]]`): transitions list processed after unrolling
                """
                if self._unroll_len == 1:
                    return data
                else:
                    # cut data into pieces whose length is unroll_len
                    split_data, residual = list_split(data, step=self._unroll_len)

                    def null_padding():
                        template = copy.deepcopy(residual[0])
                        template['done'] = True
                        template['reward'] = torch.zeros_like(template['reward'])
                        null_data = [self._get_null_transition(template) for _ in range(miss_num)]
                        return null_data

                    if residual is not None:
                        miss_num = self._unroll_len - len(residual)
                        if self._last_fn_type == 'drop':
                            # drop the residual part
                            pass
                        elif self._last_fn_type == 'last':
                            if len(split_data) > 0:
                                # copy last datas from split_data's last element, and insert in front of residual
                                last_data = copy.deepcopy(split_data[-1][-miss_num:])
                                split_data.append(last_data + residual)
                            else:
                                # get null transitions using ``null_padding``, and insert behind residual
                                null_data = null_padding()
                                split_data.append(residual + null_data)
                        elif self._last_fn_type == 'null_padding':
                            # same to the case of 'last' type and split_data is empty
                            null_data = null_padding()
                            split_data.append(residual + null_data)
                    # collate unroll_len dicts according to keys
                    if len(split_data) > 0:
                        split_data = [lists_to_dicts(d, recursive=True) for d in split_data]
                    return split_data
    
    
    对于 ``BaseSerialCollector`` 来说，``get_traj`` 方法和 ``get_train_sampler`` 方法对于大部分算法来说都是需要被调用的，因此在如下的 ``CommonPolicy`` 的代码中，两个方法都被调用了:

        .. code:: python

            # in collector

            train_sample = self._policy.get_train_sample(self._traj_cache[env_id])


            # in CommonPolicy
            def _get_train_sample(self, traj_cache: deque) -> Union[None, List[Any]]:
                # adder is defined in _init_collect
                data = self._adder.get_traj(traj_cache, self._traj_len)
                return self._adder.get_train_sample(data)
    
    3. ``get_nstep_return_data`` : 该方法同样在 ``BaseSerialCollector`` 中通过 ``Policy`` 类的 ``collect_mode.get_train_sample`` 调用，用于需要多个timestep进行计算的，如在 ``r2d2`` 算法中的调用如下：

        .. code:: python

            # in nervex/policy/r2d2.py
            # r2d2
            def _get_train_sample(self, traj_cache: deque) -> Union[None, List[Any]]:
                r"""
                Overview:
                    Get the trajectory and the n step return data, then sample from the n_step return data

                Arguments:
                    - traj_cache (:obj:`deque`): The trajectory's cache

                Returns:
                    - samples (:obj:`dict`): The training samples generated
                """
                data = self._adder.get_traj(traj_cache, self._traj_len, return_num=self._collect_burnin_step)
                data = self._adder.get_nstep_return_data(data, self._collect_nstep, self._traj_len) # call the get_nstep_return_data since we need multi timestep
                return self._adder.get_train_sample(data)

    
    该方法的具体实现代码如下:

        .. code:: python

            def get_nstep_return_data(self, data: deque, nstep: int) -> deque:
                """
                Overview:
                    Process raw traj data by updating keys ['next_obs', 'reward', 'done'] in data's dict element.
                Arguments:
                    - data (:obj:`deque`): transitions list, each element is a transition dict
                    - nstep (:obj:`int`): number of steps. If equals to 1, return ``data`` directly; \
                        Otherwise update with nstep value
                Returns:
                    - data (:obj:`deque`): transitions list like input one, but each element updated with \
                        nstep value
                """
                if nstep == 1:
                    return data
                fake_reward = torch.zeros(1)
                next_obs_flag = 'next_obs' in data[0]
                for i in range(len(data) - nstep):
                    # update keys ['next_obs', 'reward', 'done'] with their n-step value
                    if next_obs_flag:
                        data[i]['next_obs'] = copy.deepcopy(data[i + nstep]['obs'])
                    data[i]['reward'] = torch.cat([data[i + j]['reward'] for j in range(nstep)])
                    data[i]['done'] = data[i + nstep - 1]['done']
                for i in range(max(0, len(data) - nstep), len(data)):
                    if next_obs_flag:
                        data[i]['next_obs'] = copy.deepcopy(data[-1]['next_obs'])
                    data[i]['reward'] = torch.cat(
                        [data[i + j]['reward']
                        for j in range(len(data) - i)] + [fake_reward for _ in range(nstep - (len(data) - i))]
                    )
                    data[i]['done'] = data[-1]['done']
                return data
    
    4. ``get_gae`` 和 ``get_gae_with_default_last_value`` : 这两个方法用于获得序列的GAE advantage值，有关GAE的介绍请见 `RLwarmup <../rl_warmup/algorithm/rl-algo.html>`_ 。 GAE如在 ``a2c`` 算法和 ``ppo`` 算法中会被用到:

        .. code:: python

            # in nervex/policy/ppo.py
            # ppo
            def _get_train_sample(self, traj_cache: deque) -> Union[None, List[Any]]:
                r"""
                Overview:
                    Get the trajectory and calculate GAE, return one data to cache for next time calculation
                Arguments:
                    - traj_cache (:obj:`deque`): The trajectory's cache
                Returns:
                    - samples (:obj:`dict`): The training samples generated
                """
                # adder is defined in _init_collect
                data = self._adder.get_traj(traj_cache, self._traj_len, return_num=1)
                if self._traj_len == float('inf'):
                    assert data[-1]['done'], "episode must be terminated by done=True"
                data = self._adder.get_gae_with_default_last_value(
                    data, data[-1]['done'], gamma=self._gamma, gae_lambda=self._gae_lambda
                )
                return self._adder.get_train_sample(data)

            # in nervex/policy/a2c.py
            # a2c
            def _get_train_sample(self, traj: deque) -> Union[None, List[Any]]:
                r"""
                Overview:
                    Get the trajectory and the n step return data, then sample from the n_step return data
                Arguments:
                    - traj (:obj:`deque`): The trajectory's cache
                Returns:
                    - samples (:obj:`dict`): The training samples generated
                """
                # adder is defined in _init_collect
                data = self._adder.get_traj(traj, self._traj_len, return_num=1)
                if self._traj_len == float('inf'):
                    assert data[-1]['done'], "episode must be terminated by done=True"
                data = self._adder.get_gae_with_default_last_value(
                    data, data[-1]['done'], gamma=self._gamma, gae_lambda=self._gae_lambda
                )
                if self._collect_use_nstep_return:
                    data = self._adder.get_nstep_return_data(data, self._collect_nstep, self._traj_len)
                return self._adder.get_train_sample(data)
    
    该方法的具体实现代码如下:

        .. code:: python

            def get_gae(self, data: List[Dict[str, Any]], last_value: torch.Tensor, gamma: float,
                                                gae_lambda: float) -> List[Dict[str, Any]]:
                """
                Overview:
                    Get GAE advantage for stacked transitions(T timestep, 1 batch). Call ``gae`` for calculation.
                Arguments:
                    - data (:obj:`list`): transitions list, each element is a transition dict with at least ['value', 'reward']
                    - last_value (:obj:`torch.Tensor`): the last value(i.e.: the T+1 timestep)
                    - gamma (:obj:`float`): the future discount factor
                    - gae_lambda (:obj:`float`): gae lambda parameter
                Returns:
                    - data (:obj:`list`): transitions list like input one, but each element owns extra advantage key 'adv'
                """
                value = torch.stack([d['value'] for d in data] + [last_value])
                reward = torch.stack([d['reward'] for d in data])
                if self._use_cuda:
                    value = value.cuda()
                    reward = reward.cuda()
                adv = gae(gae_data(value, reward), gamma, gae_lambda)
                if self._use_cuda:
                    adv = adv.cpu()
                for i in range(len(data)):
                    data[i]['adv'] = adv[i]
                return data

            def get_gae_with_default_last_value(self, data: List[Dict[str, Any]], done: bool, gamma: float,
                                                gae_lambda: float) -> List[Dict[str, Any]]:
                """
                Overview:
                    Like ``get_gae`` above to get GAE advantage for stacked transitions. However, this function is designed in
                    case ``last_value`` is not passed. If transition is not done yet, it wouold assign last value in ``data``
                    as ``last_value``, discard the last element in ``data``(i.e. len(data) would decrease by 1), and then call
                    ``get_gae``. Otherwise it would make ``last_value`` equal to 0.
                Arguments:
                    - data (:obj:`List[Dict[str, Any]]`): transitions list, each element is a transition dict with \
                        at least['value', 'reward']
                    - done (:obj:`bool`): whether the transition reaches the end of an episode(i.e. whether the env is done)
                    - gamma (:obj:`float`): the future discount factor
                    - gae_lambda (:obj:`float`): gae lambda parameter
                Returns:
                    - data (:obj:`List[Dict[str, Any]]`): transitions list like input one, but each element owns \
                        extra advantage key 'adv'
                """
                if done:
                    last_value = torch.zeros(1)
                else:
                    last_value = data[-1]['value']
                    data = data[:-1]
                return self.get_gae(data, last_value, gamma, gae_lambda)
    