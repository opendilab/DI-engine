How to migrate your own environment to DI-engine
==============================================================

``DI-zoo`` provides users with a large number of commonly used environments for reinforcement learning（ `supported environments <https://github.com/opendilab/DI-engine#environment-versatility>`_ ），but in many research and engineering scenarios, users still need to implement an environment by themselves, and expect to quickly migrate it to ``DI-engine`` to meet the relevant specifications of ``DI-engine``. Therefore, in this section, we will introduce how to perform the above migration step by step to meet the specification of the environment base class  ``BaseEnv``  of  ``DI-engine`` , so that it can be easily applied in the training pipeline.

The following introduction will start with **Basic** and **Advanced** . **Basic** describes the functions that must be implemented, and the details that you should pay attention to ; **Advanced** describes some extended functions.

Then ``DingEnvWrapper`` will be introduced , it is a "tool" that can quickly convert simple environments such as ClassicControl, Box2d, Atari, Mujoco, GymHybrid, etc. into environments that conform to ``BaseEnv``. And there is a Q & A at the end.

Basic
~~~~~~~~~~~~~~

This section describes the specification constraints that users **MUST** meet, and the features that must be implemented when migrating environments.

If you want to use the environment in the DI-engine, you need to implement a subclass environment that inherits from  ``BaseEnv``, such as  ``YourEnv``. The relationship between  ``YourEnv``  and your own environment is a `composition <https://en.wikipedia.org/wiki/Object_composition>`_ relationship, that is, within a  ``YourEnv`` instance, there will be an instance of an environment that is native to the user (eg, a gym-type environment).

Reinforcement learning environments have some common major interfaces that are implemented by most environments, such as ``reset()``, ``step()``, ``seed()``, etc. In DI-engine, ``BaseEnv`` will further encapsulate these interfaces. In most cases, Atari will be used as an example to illustrate. For specific code, please refer to `Atari Env <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/envs/atari_env.py>`_  and  `Atari Env Wrapper <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/envs/atari_wrappers.py>`_


1. ``__init__()``

   In general, the environment may be instantiated in the ``__init__`` method, **but** in DI-engine, in order to facilitate the support of "environment vectorization" modules like ``EnvManager`` , the environment instances generally use the **Lazy Init** mechanism, that is, the ``__init__`` method does not initialize the real original environment instance, but only sets the relevant **parameter configuration**. When the ``reset`` method is called for the first time , the actual environment initialization will take place.

   Take Atari for example. ``__init__`` does not instantiate the environment, it just sets the parameter configuration value ``self._cfg``, and initializes the variable ``self._init_flag`` to ``False`` (indicating that the environment has not been instantiated).


   .. code:: python
      
      class AtariEnv(BaseEnv):

         def __init__(self, cfg: dict) -> None:
            self._cfg = cfg
            self._init_flag = False

2. ``seed()``

   ``seed`` is used to set the random seed in the environment. There are two types of the random seed in the environment that need to be set, one is the random seed of the **original environment**, the other is the library seeds (e.g. ``random`` , ``np.random``, etc.) in various **environment transformations**.

   For the second type, the setting of the seed of the random library is relatively simple, and it is set directly in the ``seed`` method of the environment.

   For the first type, the seed of the original environment is only assigned in the ``seed`` method, but not really set; the real setting is inside the ``reset`` method of the calling environment, the specific original environment ``reset`` before setting.

   .. code:: python

      class AtariEnv(BaseEnv):
         
         def seed(self, seed: int, dynamic_seed: bool = True) -> None:
            self._seed = seed
            self._dynamic_seed = dynamic_seed
            np.random.seed(self._seed)

   For the seeds of the original environment, DI-engine has the concepts of **static seeds** and **dynamic seeds**.
   
   **Static seed** is used in the test environment (evaluator_env) to ensure that the random seed of all episodes are the same, that is, only the fixed static seed value of ``self._seed`` is used when ``reset``. Need to manually pass the ``dynamic_seed`` parameter to ``False`` in the ``seed`` method.

   **Dynamic seed** is used for the training environment (collector_env), try to make the random seed of each episode different, that is, when ``reset``, a random number generator will be used ``100 * np.random.randint(1, 1000)`` (but the seed of this random number generator is fixed by the environment's ``seed`` method, so the reproducibility of the experiment is guaranteed). You need to manually pass in the ``dynamic_seed`` parameter as ``True`` in ``seed`` (or you can not pass it, because the default parameter is ``True``).

3. ``reset()``

   The **Lazy Init** initialization method of DI-engine has been introduced in the ``__init__`` method, that is, the actual environment initialization is performed when **the first call** ``reset`` method is performed.

   The ``reset`` method will judge whether the actual environment needs to be instantiated according to ``self._init_flag`` (if it is ``False``, it will be instantiated; otherwise, it has already been instantiated and can be used directly), and Set the random seed, then call the ``reset`` method of the original environment to get the observation value ``obs`` in the initial state, and convert it to the ``np.ndarray`` data format (will be explained in detail in 4) , and initialize the value of ``self._final_eval_reward`` (will be explained in detail in 5), in Atari ``self._final_eval_reward`` refers to the cumulative sum of the real rewards obtained by a whole episode, used to evaluate the agent Performance on this environment, not used for training.

   .. code:: python
      
      class AtariEnv(BaseEnv):

         def __init__(self, cfg: dict) -> None:
            self._cfg = cfg
            self._init_flag = False

         def reset(self) -> np.ndarray:
            if not self._init_flag:
               self._env = self._make_env(only_info=False)
               self._init_flag = True
            if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
               np_seed = 100 * np.random.randint(1, 1000)
               self._env.seed(self._seed + np_seed)
            elif hasattr(self, '_seed'):
               self._env.seed(self._seed)
            obs = self._env.reset()
            obs = to_ndarray(obs)
            self._final_eval_reward = 0.
            return obs

4. ``step()``

   The ``step`` method is responsible for receiving the ``action`` of the current timestep, and then giving the ``reward`` of the current timestep and the ``obs`` of the next timestep. In DI-engine, you also need to give: The flag ``done`` of whether the current episode ends (here requires ``done`` must be of type ``bool``, not ``np.bool``), other information in the form of a dictionary ``info`` (which includes at least the key ``self._final_eval_reward``).

   After getting ``reward`` , ``obs`` , ``done`` , ``info`` and other data, it needs to be processed and converted into ``np.ndarray`` format to conform to the DI-engine specification. ``self._final_eval_reward`` will accumulate the actual reward obtained at the current step at each time step, and return the accumulated value at the end of an episode ( ``done == True``).

   Finally, put the above four data into ``BaseEnvTimestep`` defined as ``namedtuple`` and return (defined as: ``BaseEnvTimestep = namedtuple('BaseEnvTimestep', ['obs', 'reward', 'done ', 'info'])`` )
   
   .. code:: python

      from ding.envs import BaseEnvTimestep

      class AtariEnv(BaseEnv):
         
         def step(self, action: np.ndarray) -> BaseEnvTimestep:
            assert isinstance(action, np.ndarray), type(action)
            action = action.item()
            obs, rew, done, info = self._env.step(action)
            self._final_eval_reward += rew
            obs = to_ndarray(obs)
            rew = to_ndarray([rew])  # Transformed to an array with shape (1, )
            if done:
               info['final_eval_reward'] = self._final_eval_reward
            return BaseEnvTimestep(obs, rew, done, info)

5. ``self._final_eval_reward``

   In the Atari environment, ``self._final_eval_reward`` refers to the cumulative sum of all rewards of an episode, and the data type of ``self._final_eval_reward`` must be a python native type, not ``np.array``.

      - In the ``reset`` method, set the current ``self._final_eval_reward`` to 0;
      - In the ``step`` method, add the actual reward obtained at each time step to ``self._final_eval_reward``.
      - In the ``step`` method, if the current episode has ended ( ``done == True`` ), then add to the ``info`` dictionary and return: ``info['final_eval_reward'] = self._final_eval_reward``

   However, other environments may not require the sum of ``self._final_eval_reward``. For example, in smac, the winning percentage of the current episode is required, so it is necessary to modify the simple accumulation in the second step ``step`` method. Instead, we should record the game situation and finally return the calculated winning percentage at the end of the episode.

6. Data Specifications

   DI-engine requires that the input and output data of each method in the environment must be in ``np.ndarray`` format, and the data dtype must be ``np.int64`` (integer), ``np.float32`` ( float) or ``np.uint8`` (image). include:

      -  ``obs`` returned by the ``reset`` method
      -  ``action`` received by the ``step`` method
      -  ``obs`` returned by the ``step`` method
      -  ``reward`` returned by the ``step`` method, here also requires that ``reward`` must be **one-dimensional**, not zero-dimensional, for example, Atari will expand zero-dimensional to one-dimensional ``rew = to_ndarray([rew])``
      -  ``done`` returned by the ``step`` method must be of type ``bool``, not ``np.bool``


Advanced
~~~~~~~~~~~~

1. Environment preprocessing wrapper

   If many environments are to be used in reinforcement learning training, some preprocessing is required to achieve the purpose of increasing randomness, data normalization, and ease of training. These preprocessing are implemented in the form of wrappers (for the introduction of wrappers, please refer to `here <./env_wrapper_zh.html>`_ ）.
   
   Each wrapper for environment preprocessing is a subclass of ``gym.Wrapper``. For example, ``NoopResetEnv`` is to perform a random number of No-Operation actions at the beginning of the episode. It is a means of increasing randomness. It is used as follows:
   
   .. code:: python
      
      env = gym.make('Pong-v4')
      env = NoopResetEnv(env)
   
   Since the ``reset`` method is implemented in ``NoopResetEnv``, the corresponding logic in ``NoopResetEnv`` will be executed when ``env.reset()``.

   The following env wrapper has been implemented in DI-engine:( in ``ding/envs/env_wrappers/env_wrappers.py``)

      - ``NoopResetEnv``: perform a random number of No-Operation actions at the beginning of the episode
      - ``MaxAndSkipEnv``: Returns the maximum value in several frames, which can be considered as a kind of max pooling on time steps
      - ``WarpFrame``: Convert the original image to the color code using ``cvtColor`` of the ``cv2`` library, and resize it into an image of a certain length and width (usually 84x84)
      - ``ScaledFloatFrame``: normalize the observation to the interval [0, 1] (keep the dtype as ``np.float32``)
      - ``ClipRewardEnv``: Pass the reward through a symbolic function to ``{+1, 0, -1}``
      - ``FrameStack``: stacks a certain number (usually 4) of frames together as a new observation, which can be used to deal with POMDP situations, for example, the speed direction of the movement cannot be known by a single frame of information
      - ``ObsTransposeWrapper``: Transpose observation to put channel to first dim
      - ``ObsNormEnv``: use ``RunningMeanStd`` to normalize the observation for sliding windows
      - ``RewardNormEnv``: use ``RunningMeanStd`` to normalize the reward with sliding window
      - ``RamWrapper``: Wrap ram env into image-like env
      - ``EpisodicLifeEnv``: treat environments with multiple lives built in (eg Qbert), and treat each life as an episode
      - ``FireResetEnv``: execute action 1 (fire) immediately after environment reset
      - ``GymHybridDictActionWrapper``: Transform Gym-Hybrid's original ``gym.spaces.Tuple`` action space to ``gym.spaces.Dict``

   If the above wrappers cannot meet your needs, you can also customize the wrappers yourself.

   It is worth mentioning that each wrapper must not only complete the change of the corresponding observation/action/reward value, but also modify its space accordingly (if and only when shpae, dtype, etc. are modified), this method will be described in the next described in detail in the section.

2. Three space attributes ``observation/action/reward space``

   If you want to automatically create a neural network based on the dimensions of the environment, or use the ``shared_memory`` technique in the ``EnvManager`` to speed up the transmission of large tensor data returned by the environment, you need to let the environment support provide the attribute  ``observation_space`` ``action_space`` ``reward_space``  .

   .. note::
      
      For the sake of code extensibility, we **strongly recommend implementing these three space attributes**.
   
   The spaces here are all instances of subclasses of ``gym.spaces.Space``, the most commonly used ``gym.spaces.Space`` include ``Discrete`` ``Box`` ``Tuple`` ``Dict``  etc. **shape** and **dtype** need to be given in space. In the original gym environment, most of them will support ``observation_space``, ``action_space`` and ``reward_range``. In DI-engine, ``reward_range`` is also expanded into ``reward_space``, so that this All three remain the same.

   For example, here are the three attributes of cartpole:

   .. code:: python

      class CartpoleEnv(BaseEnv):
         
         def __init__(self, cfg: dict = {}) -> None:
            self._observation_space = gym.spaces.Box(
                  low=np.array([-4.8, float("-inf"), -0.42, float("-inf")]),
                  high=np.array([4.8, float("inf"), 0.42, float("inf")]),
                  shape=(4, ),
                  dtype=np.float32
            )
            self._action_space = gym.spaces.Discrete(2)
            self._reward_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1, ), dtype=np.float32)

         @property
         def observation_space(self) -> gym.spaces.Space:
            return self._observation_space

         @property
         def action_space(self) -> gym.spaces.Space:
            return self._action_space

         @property
         def reward_space(self) -> gym.spaces.Space:
            return self._reward_space

   Since the cartpole does not use any wrapper, its three spaces are fixed. However, if an environment like Atari has been decorated with multiple wrappers, it is necessary to modify the corresponding space after each wrapper wraps the original environment. For example, Atari will use ``ScaledFloatFrameWrapper`` to normalize the observation to the interval [0, 1], then it will modify its ``observation_space`` accordingly:

   .. code:: python

      class ScaledFloatFrameWrapper(gym.ObservationWrapper):
         
         def __init__(self, env):
            # ...
            self.observation_space = gym.spaces.Box(low=0., high=1., shape=env.observation_space.shape, dtype=np.float32)


3. ``enable_save_replay()``

   ``DI-engine`` does not require the implementation of the ``render`` method. If you want to complete the visualization, we recommend implementing the ``enable_save_replay`` method to save the game video.
   
   This method is called before the ``reset`` method and after the ``seed`` method, in which the path to the recording storage is specified. It should be noted that this method **does not directly store the video**, but only sets a flag for whether to save the video. The code and logic for actually storing the video needs to be implemented by yourself. (Because multiple environments may be opened, and each environment runs multiple episodes, it needs to be distinguished in the file name)

   Here, an example in DI-engine is given. The ``reset`` method uses the decorator provided by ``gym`` to encapsulate the environment, giving it the function of storing game videos, as shown in the code:

   .. code:: python

      class AtariEnv(BaseEnv):

         def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
            if replay_path is None:
               replay_path = './video'
            self._replay_path = replay_path

         def reset():
            # ...
            if self._replay_path is not None:
               self._env = gym.wrappers.RecordVideo(
                  self._env,
                  video_folder=self._replay_path,
                  episode_trigger=lambda episode_id: True,
                  name_prefix='rl-video-{}'.format(id(self))
               )
            # ...
   
   In actual use, the order of calling these methods should be:

   .. code:: python
      
      atari_env = AtariEnv(easydict_cfg)
      atari_env.seed(413)
      atari_env.enable_save_replay('./replay_video')
      obs = atari_env.reset()
      # ...


4. Use different config for training environment and test environment

   The environment used for training (collector_env) and the environment used for testing (evaluator_env) may use different configuration items. You can implement a static method in the environment to implement custom configuration for different environment configuration items. Take Atari as an example:

   .. code:: python

      class AtariEnv(BaseEnv):

         @staticmethod
         def create_collector_env_cfg(cfg: dict) -> List[dict]:
            collector_env_num = cfg.pop('collector_env_num')
            cfg = copy.deepcopy(cfg)
            cfg.is_train = True
            return [cfg for _ in range(collector_env_num)]

         @staticmethod
         def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
            evaluator_env_num = cfg.pop('evaluator_env_num')
            cfg = copy.deepcopy(cfg)
            cfg.is_train = False
            return [cfg for _ in range(evaluator_env_num)]

   In actual use, the original configuration item ``cfg`` can be converted to obtain two versions of configuration items for training and testing:

   .. code:: python

      # env_fn is an env class
      collector_env_cfg = env_fn.create_collector_env_cfg(cfg)
      evaluator_env_cfg = env_fn.create_evaluator_env_cfg(cfg)

   Setting the ``cfg.is_train`` item will use different decorations in the wrapper accordingly. For example, if ``cfg.is_train == True``, a symbolic function of reward will be used to map to ``{+1, 0, -1}`` to facilitate training, if ``cfg.is_train == False`` Then the original reward value will remain unchanged, which is convenient for evaluating the performance of the agent during testing.

5. ``random_action()``

   Some off-policy algorithms hope to use a random strategy to collect some data to fill the buffer before training starts, and complete the initialization of the buffer. For such a need, DI-engine encourages the implementation of the ``random_action`` method.

   Since the environment already implements ``action_space``, you can directly call the ``Space.sample()`` method provided in the gym to randomly select actions. But it should be noted that since DI-engine requires all returned actions to be in ``np.ndarray`` format, some necessary dtype conversions may be required. The ``int`` and ``dict`` types are converted to the ``np.ndarray`` type using the ``to_ndarray`` function, as shown in the following code:

   .. code:: python

      def random_action(self) -> np.ndarray:
         random_action = self.action_space.sample()
         if isinstance(random_action, np.ndarray):
               pass
         elif isinstance(random_action, int):
               random_action = to_ndarray([random_action], dtype=np.int64)
         elif isinstance(random_action, dict):
               random_action = to_ndarray(random_action)
         else:
               raise TypeError(
                  '`random_action` should be either int/np.ndarray or dict of int/np.ndarray, but get {}: {}'.format(
                     type(random_action), random_action
                  )
               )
         return random_action

6. ``default_config()``

   If an environment has some default or commonly used configuration items, you can consider setting the class variable ``config`` as **default config** (for the convenience of external access, you can also implement the class method ``default_config``, which returns config). As shown in the following code:
   
   When running an experiment, a **user config** file for this experiment is configured, such as ``dizoo/mujoco/config/ant_ddpg_config.py``. In the user config file, you can omit this part of the key-value pair, and merge **default config** with **user config** through ``deep_merge_dicts`` (remember to use the default config as the first parameter here, the user config is used as the second parameter to ensure that the user config has a higher priority). As shown in the following code:
   
   .. code:: python
      
      class MujocoEnv(BaseEnv):

         @classmethod
         def default_config(cls: type) -> EasyDict:
            cfg = EasyDict(copy.deepcopy(cls.config))
            cfg.cfg_type = cls.__name__ + 'Dict'
            return cfg

         config = dict(
            use_act_scale=False,
            delay_reward_step=0,
         )

         def __init__(self, cfg) -> None:
            self._cfg = deep_merge_dicts(self.config, cfg)


7. Environment implementation correctness check

   We provide a set of inspection tools for user-implemented environments to check:
  
   - data type of observation/action/reward
   - reset/step method
   - Whether there are unreasonable identical references in the observation of two adjacent time steps (that is, deepcopy should be used to avoid identical references)
   
   The implementation of the check tool is in ``ding/envs/env/env_implementation_check.py`` .
   For the usage of the check tool, please refer to ``ding/envs/env/tests/test_env_implementation_check.py`` 's ``test_an_implemented_env``。



DingEnvWrapper
~~~~~~~~~~~~~~~~~~~~~~~~

``DingEnvWrapper`` can quickly convert simple environments such as ClassicControl, Box2d, Atari, Mujoco, GymHybrid, etc., to ``BaseEnv`` compliant environments.

Note: The specific implementation of ``DingEnvWrapper`` can be found in ``ding/envs/env/ding_env_wrapper.py``, in addition, you can see `Example <https://github.com/opendilab/DI-engine/blob/main/ding/envs/env/tests/test_ding_env_wrapper.py>`_ for more info.



Q & A
~~~~~~~~~~~~~~

1. How should the MARL environment be migrated?
   
   You can refer to `Competitive RL <../env_tutorial/competitive_rl_zh.html>`_ 

   - If the environment supports both single-agent, double-agent or even multi-agent, consider different mode classifications
   - In a multi-agent environment, the number of action and observation matches the number of agents, but the reward and done are not necessarily the same. It is necessary to clarify the definition of reward
   - Note how the original environment requires actions and observations to be combined (tuples, lists, dictionaries, stacked arrays and so on)


2. How should the environment of the hybrid action space be migrated?
   
   You can refer to  `Gym-Hybrid <../env_tutorial/gym_hybrid_zh.html>`_

   - Some discrete actions (Accelerate, Turn) in Gym-Hybrid need to give corresponding 1-dimensional continuous parameters to represent acceleration and rotation angle, so similar environments need to focus on the definition of their action space
