Random seed
=============

In reinforcement learning, the seed of random number
has an influence on the result of the algorithm.
To replicate other people's experiments, we
need to use the same random seed.

Random seed in Nervex
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Firstly, in the each entry function, we have a global
random ``seed`` parameter. For example,
in ``nervex/entry/serial_entry.py``,

.. code:: python

    def serial_pipeline(..., seed: int = 0, ... ):
        ...

In ``nervex/utils/default_helper.py``, we define
a ``set_pkg_seed`` function called in
the entry function to set the seed of all used package.


.. code:: python

    def set_pkg_seed(seed: int, use_cuda: bool = True) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if use_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

For collect or evaluator envs, If only one seed is given,
nervex will generate an increasing list of random seeds as the seeds for that set of envs.

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

To make env more diversity, nerveX also support ``dynamic_seed`` for each concrete env.
As in ``nervex/envs/env/nervex_env_wrapper.py``, for each env, we add a random number to the original seed.

.. code:: python

    def reset(self) -> None:
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        ...


.. tip::
    When using multiple processes, the random seed
    of the child process will not inherit the seed of the
    parent process and will remain the system default seed.
    As shown in ``app_zoo/atari/envs/atari_env.py#L156``,
    we solve this problem by resetting the seeds in each env.

    .. code:: python

        def seed(self, seed: int, dynamic_seed: bool = True) -> None:
            ...
            np.random.seed(self._seed)
