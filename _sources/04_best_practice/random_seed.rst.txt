Random Seed
=============

In reinforcement learning, different random numbers as seeds will affect the results of the algorithm. In order to reproduce \
or fairly compare experimental results of other algorithms, we need to use the same random seeds.


First, in each entry function, we have a global random "seed" parameter. For example, in ``ding/entry/serial_entry.py`` ,

.. code:: python

    def serial_pipeline(..., seed: int = 0, ...):
        ...

In ``ding/utils/default_helper.py`` , we define a ``set_pkg_seed`` function that is called at the beginning \
of the entry function (as shown below) to set "seed" of all associated packages used.


.. code:: python

    def set_pkg_seed(seed: int, use_cuda: bool = True) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if use_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

For a collector or evaluator, if only one seed is given, DI-engine will generate a list of random seeds \
for the set of environments in order to give each subenvironment different randomness.

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

To make the environment more diverse, DI-engine also allows ``dynamic_seed`` to be enabled when many episodes are run per environment.

The selection of random seeds is critical to the environment. For example, in the environmental LunarLander, the randomness of the environment is jointly determined by the randomness of the initial landing moment and the randomness in the landing process.
For the initial moment, the lander starts from the top center of the screen, and a random initial force is applied to its center of mass to ensure that the horizontal velocity, vertical velocity, \
angle and angular velocity are different in the initial state under different seeds. \
At the same time, the distribution of lunar terrain will be determined according to the random number sampled.
For the landing process, there is a Stochastic dispersion Force (Stochastic dispersion) in the lander's dynamics equation to ensure that environmental transfer functions vary among different seeds.

As the link ``ding/envs/env/DI-engine_env_wrapper.py`` shows, first, DI-engine sets the environment seed when resetting the environment, 
And if ``dynamic_seed`` is True, DI-engine adds a random integer to the original seed to make each episode different. And reproducibility can be guaranteed by seeding this random generator. The random generator is usually ``numpy.random``.

By default, we enable dynamic_seed for data collection and disable it for estimation. This has the advantage of allowing us to collect more diverse training data, improving the final performance, but may reduce the convergence rate.
Turning off dynamic_seed during evaluation ensures that each evaluation strategy is evaluated on the same set of random seeds, increasing the comparability of model evaluation results over different training steps.

.. code:: python

    def reset(self) -> None:
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        ...


.. tip::
     
     When multiple processes are used, the child's random seed does not inherit the parent's random seed, 
     but instead remains the system's default seed.
     As shown `here <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/envs/atari_env.py#L49>`_ ,
     we solved this problem by resetting the relevant seeds in each environment.
     

    .. code:: python

        def seed(self, seed: int, dynamic_seed: bool = True) -> None:
            ...
            np.random.seed(self._seed)
