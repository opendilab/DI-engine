随机种子
=============

在强化学习中，不同的随机数作为种子对算法的结果会有影响。为了复现或者公平比较其他算法的实验结果，我们需要使用相同的随机种子。


首先，在每个入口函数中，我们有一个全局的随机“种子”参数。例如在 ``ding/entry/serial_entry.py`` 中,

.. code:: python

    def serial_pipeline(..., seed: int = 0, ...):
        ...

在 ``ding/utils/default_helper.py`` 中, 我们定义了一个在入口函数中被调用的 ``set_pkg_seed`` 函数（如下图），以便于
为所有使用的相关程序包设置种子。


.. code:: python

    def set_pkg_seed(seed: int, use_cuda: bool = True) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if use_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

对于 collector 或 evaluator的环境，如果只给出一个种子， DI-engine 将生成一个随机种子列表作为该组环境的种子。

.. code:: python

    def seed(self, seed: Union[List[int], int], dynamic_seed: bool = None) -> None:
        """
        Overview:
            Set the seed for each environment.
        Arguments:
            - seed (:obj:`Union[List[int], int]`): List of seeds for each environment, \
                or one seed for the first environment and other seeds are generated automatically.
        """
        if isinstance(seed, numbers.Integral):
            seed = [seed + i for i in range(self.env_num)]
        elif isinstance(seed, list):
            assert len(seed) == self._env_num, "len(seed) {:d} != env_num {:d}".format(len(seed), self._env_num)
            seed = seed
        self._env_seed = seed
        self._env_dynamic_seed = dynamic_seed

为了让环境更加多样化, DI-engine 也支持在每一个环境跑很多个episode的时候启用 ``dynamic_seed``。
如链接所示的那样 ``ding/envs/env/DI-engine_env_wrapper.py``, 首先，DI-engine 在重置环境时设置环境种子， 并且如果 ``dynamic_seed`` 是 True, DI-engine 会在原始种子中添加一个随机整数，以使每个
episode 不同。 并且可以通过设置这个随机生成器的种子来保证重现性。 这个随机生成器一般是 ``numpy.random``。

默认情况下，我们在收集数据时启用 dynamic_seed，在评估时禁用它。这样做的好处是可以让我们收集更多样化的训练数据，提高最终性能，但可能会降低收敛速度。
在评估时关闭 dynamic_seed 可以保证每次评估策略时，是在一组相同的随机种子上评估的，增加不同训练步数上模型评估结果的可比性。

.. code:: python

    def reset(self) -> None:
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        ...


.. tip::
     
     当使用多个进程时，子进程的随机种子不会继承
     父进程的随机种子，而是保持系统的默认种子。
     如 `这里 <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/envs/atari_env.py#L49>`_ 所示，
     我们通过重置每个环境中的相关种子来解决这个问题。
     

    .. code:: python

        def seed(self, seed: int, dynamic_seed: bool = True) -> None:
            ...
            np.random.seed(self._seed)
