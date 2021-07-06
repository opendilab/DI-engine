Random seed
=============

In reinforcement learning, the seed of random number
has an influence on the result of the algorithm.
To replicate other people's experiments, we
need to use the same random seed.


Firstly, in the each entry function, we have a global
random ``seed`` parameter. For example,
in ``ding/entry/serial_entry.py``,

.. code:: python

    def serial_pipeline(..., seed: int = 0, ... ):
        ...

In ``ding/utils/default_helper.py``, we define
a ``set_pkg_seed`` function called in
the entry function to set the seed of all used package.


.. code:: python

    def set_pkg_seed(seed: int, use_cuda: bool = True) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if use_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

For collector or evaluator envs, If only one seed is given,
DI-engine will generate an increasing list of random seeds as the seeds for that set of envs.

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

To make env more diversity, DI-engine also supports ``dynamic_seed`` for each concrete env running several episodes.
As is shown in ``ding/envs/env/DI-engine_env_wrapper.py``, first, DI-engine sets env seed when it resets, and if ``dynamic_seed`` is True, DI-engine would add a random integer number to the original seed to make each
episode different. And the reproducibility can be ensured by setting the seed of this random generator(usually numpy.random).

By default, we enable dynamic_seed when collecting data while disable it in evaluation, in order to collect more diverse training data, which could improve final performance but slow down converge speed a bit and 
keep evaluation consistency.

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
    As is shown in ``dizoo/atari/envs/atari_env.py#L156``,
    we solve this problem by resetting the related seeds in each env.
    Please care more about this.

    .. code:: python

        def seed(self, seed: int, dynamic_seed: bool = True) -> None:
            ...
            np.random.seed(self._seed)
