How to migrate your environment to DI-engine
==================================================

Although a large number of commonly used Reinforcement Learning environments have been provided in ``DI-zoo`` (`DI-engine supported environments <https://github.com/opendilab/DI-engine#environment-versatility>`_ ), you may still need to migrate your environment to ``DI-engine``. Therefore, in this section, we will introduce how to perform the above migration step by step to meet the specifications of the ``DI-engine'' basic environment base class ``BaseEnv``, so that it can be easily applied in the training pipeline.

The following introduction will be divided into two parts: **Basic** and **Advanced**. **Basic** indicates the functions that must be implemented and details that must be noticed if you want to run the pipeline correctly; **Advanced** indicates some expanded functions.

Basic
~~~~~~~~~~~~~~

This section will introduce the specification constraints that users must obey and the functions that must be implemented when migrating the environment.

If you want to use the environment in DI-engine, you need to implement a subclass environment derived from ``BaseEnv``, such as ``YourEnv``. The relationship between ``YourEnv`` and your own environment is a `combination <https://www.cnblogs.com/chinxi/p/7349768.html>`_ relationship, that is, a ``YourEnv`` instance will hold an instance of your own original environment.

The Reinforcement Learning environment has some major interfaces commonly implemented by most environments, such as ``reset()``, ``step()``, ``seed()``, etc. In DI-engine, ``BaseEnv`` will further encapsulate these interfaces. In following cases, Atari will be used as an example for description. Please refer to `Atari Env <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/envs/atari_env.py>`_ and `Atari Env Wrapper <https://github. com/opendilab/DI-engine/blob/main/dizoo/atari/envs/atari_wrappers.py>`_ for code.


1. ``__init__()``

   Under normal circumstances, the environment may be instantiated in the ``__init__`` method, but in DI-engine, in order to facilitate parallel operations such as "environment vectorization" in ``EnvManager``, the environment instance generally adopts **Lazy Init** pattern, that is, the ``__init__`` method does not initialize the real original environment instance, but only sets the relevant **parameter & configuration**. The actual environment will not be initialized until the ``reset`` method is called **for the first time**.

   Take Atari as an example. ``__init__`` does not instantiate the environment, but only sets the configuration ``self._cfg``, and initialize member attribute ``self._init_flag`` which is used to record whether the ``reset`` method is called for the first time (i.e. Whether the environment has not been initialized).


   .. code:: python
      
      class AtariEnv(BaseEnv):

         def __init__(self, cfg: dict) -> None:
            self._cfg = cfg
            self._init_flag = False

2. ``seed()``

   ``seed`` is used to set the random seed in the environment. There are two types of random seeds in the environment needed to be set. One is the random seed in the **original environment**, and the other is the random seed used in various **environment transformations** (Often random seed of the libraries, e.g. ``random``, ``np.random``). If your environment does not have any randomness at all (including "original environment" and "environmental transformation"), then you don't need to implement this method.

   The setting of the seed of the random library is relatively simple. You can set it directly in ``seed`` method of the environment.

   However, the seed of the original environment is only assigned in the ``seed`` method, but not really set. The actual setting is when calling environment's ``reset`` method, and right before the specific original environment's ``reset``.

   .. code:: python

      class AtariEnv(BaseEnv):
         
         def seed(self, seed: int, dynamic_seed: bool = True) -> None:
            self._seed = seed
            self._dynamic_seed = dynamic_seed
            np.random.seed(self._seed)

   For the seeds in original environment, DI-engine defines **static seed** and **dynamic seed**.
   
   **Static seed** is used in the evaluation environment to ensure that the random seed of all episodes are the same, that is, only the fixed static seed value of ``self._seed`` will be used when ``reset``. You need to manually pass in the ``dynamic_seed`` parameter as ``False`` in the ``seed`` method.

   **Dynamic seed** is used in the training environment, trying to make the random seed of each episode different, they are all generated in the ``reset`` method by a random generator ``100 * np.random.randint(1 , 1000)`` (but the seed of this random number generator is fixed by the environmental ``seed`` method, guranteeing the reproducibility of the experiment). You need not pass the ``dynamic_seed`` parameter in the ``seed`` method, or pass the parameter as ``True``.

3. ``reset()``

   In ``__init__`` method, we have already introduced DI-engine's **Lazy Init** pattern, that is, the actual environment is not initialized untl ``reset`` method is called **for the first time**.

   ``reset`` method will judge whether the actual environment needs to be instantiated according to ``self._init_flag``, and set random seeds. Then call original environment's ``reset`` method to get the initial observation and convert to ``np.ndarray`` (will be explained in detail in 5.). Then initialize the value of ``self._final_eval_reward`` (will be explained in detail in 4.). In Atari ``self._final_eval_reward`` refers to the cumulative sum of the true rewards obtained in the entire episode. It is used to evaluate the agent's performance in the environment, not used for training.

   .. code:: python
      
      class AtariEnv(BaseEnv):

         def __init__(self, cfg: dict) -> None:
            self._cfg = cfg
            self._init_flag = False

         def reset(self) -> np.ndarray:
            if not self._init_flag:
               self._env = self._make_env(only_info=False)
               self._init_flag = True
            if hasattr(self,'_seed') and hasattr(self,'_dynamic_seed') and self._dynamic_seed:
               np_seed = 100 * np.random.randint(1, 1000)
               self._env.seed(self._seed + np_seed)
            elif hasattr(self,'_seed'):
               self._env.seed(self._seed)
            obs = self._env.reset()
            obs = to_ndarray(obs)
            self._final_eval_reward = 0.
            return obs

4. ``step()``

   ``step`` method is responsible for receiving this time ``action``, then giving this time ``reward`` and next time ``obs``. In DI-engine, you also need to give: The flag of whether current episode is finished ``done``, and other information in the form of a dictionary ``info`` (such as ``self._final_eval_reward``).

   After getting ``reward`` ``obs`` ``done`` ``info``, you need to convert them into ``np.ndarray`` format to ensure compliance with DI-engine specifications. In each time step ``self._final_eval_reward`` will accumulate the current real reward, and return the accumulated value when the episode ends (``done == True`` ).

   Finally, you should put the above four data into ``BaseEnvTimestep`` defined as ``namedtuple`` and return (defined as: ``BaseEnvTimestep = namedtuple('BaseEnvTimestep', ['obs','reward','done ','info'])``)
   
   .. code:: python

      from ding.envs import BaseEnvTimestep

      class AtariEnv(BaseEnv):
         
         def step(self, action: np.ndarray) -> BaseEnvTimestep:
            assert isinstance(action, np.ndarray), type(action)
            action = action.item()
            obs, rew, done, info = self._env.step(action)
            self._final_eval_reward += rew
            obs = to_ndarray(obs)
            rew = to_ndarray([rew]) # Transformed to an array with shape (1,)
            if done:
               info['final_eval_reward'] = self._final_eval_reward
            return BaseEnvTimestep(obs, rew, done, info)

5. ``self._final_eval_reward``

   In the Atari environment, ``self._final_eval_reward`` refers to the cumulative sum of all rewards in an episode.

      - In the ``reset`` method, set the current ``self._final_eval_reward`` to 0;
      - In the ``step`` method, add the reward obtained at each time step to ``self._final_eval_reward``.
      - In the ``step`` method, if the current episode has ended (``done == True``, here it is required that ``done`` must be of type ``bool``, not ``np.bool`` ), then add it to the ``info`` dictionary and return: ``info['final_eval_reward'] = self._final_eval_reward``

   However, in other environments, what may be needed is not the sum of the rewards in an episode. For example, in smac, the win rate is needed, so you need to modify the accumulation in ``step`` method, to recording the games' result, and finally return the calculated win rate at the end of the episode.

6. Data Specification

   In DI-engine's environment, all methods' input and output data must be ``np.ndarray``, and the dtype needs to be ``np.int64`` (integer) or ``np.float32`` ( Floating point number). Includes:

      - ``obs`` returned in ``reset`` method
      - ``action`` received in ``step`` method
      - ``obs`` returned in ``step`` method
      - ``reward`` returned in ``step`` method. Here also requires that ``reward`` must be **one-dimensional**, not zero-dimensional, such as the code in Atari ``rew = to_ndarray( [rew])``
      - ``done`` returned in ``step`` method. Must be ``bool`` type, rather than ``np.bool`` type.


Advanced
~~~~~~~~~~~~

1. Environment preprocessing wrapper

   If an environment is to be used in Reinforcement Learning training, some preprocessing is required to increase randomness, normalize data, and make training easier. These preprocessing are implemented in the form of wrapper (for the introduction of wrapper, please refer to `this <../feature/wrapper_hook_overview_zh.html#wrapper>`_ ).
   
   Each preprocessing wrapper is a subclass of ``gym.Wrapper``. For example, ``NoopResetEnv`` is to perform a random number of No-Operation actions at the very beginning of an episode. It is a means to increase randomness. THe code is:
   
   .. code:: python
      
      env = gym.make('PongNoFrameskip-v4')
      env = NoopResetEnv(env)
   
   Since the ``reset`` method is implemented in ``NoopResetEnv``, the corresponding code in ``NoopResetEnv`` will be executed when calling ``env.reset()``.

   The following env wrappers has been implemented in DI-engine: (in ``ding/envs/env_wrappers/env_wrappers.py``)

      - ``NoopResetEnv``: At the beginning of the episode, performs a random number of No-Operation actions
      - ``MaxAndSkipEnv``: Returns the maximum value in a few frames, which can be considered as a kind of max pooling over timestep
      - ``WarpFrame``: Uses the ``cvtColor`` in ``cv2`` library to convert the original image's color, then resizes the image to certain length and width (usually 84x84)
      - ``ScaledFloatFrame``: Normalizes observation to the interval [0, 1] (keeps dtype as ``np.float32``)
      - ``ClipRewardEnv``: Passes reward through a sign function to become ``{+1, 0, -1}``
      - ``FrameStack``: Stack a certain number (usually 4) of frames together as a new observation, which can be used to handle POMDP situations, for example, a single frame of information cannot know the speed direction of the movement
      - ``ObsTransposeWrapper``: Convert the image of ``(H, W, C)`` to the image of ``(C, H, W)``
      - ``ObsNormEnv``: Uses ``RunningMeanStd`` to normalize the sliding window of observation
      - ``RewardNormEnv``: Uses ``RunningMeanStd`` to normalize the sliding window of reward
      - ``RamWrapper``: Converts the observation shape of the Ram type environment to a similar image (128, 1, 1)
      - ``EpisodicLifeEnv``: Used in environments with multiple lives (such as Qbert); Regards each life as an episode
      - ``FireResetEnv``: Executes action 1 (fire) immediately after the environment is reset
      - ``GymHybridDictActionWrapper``: Transform Gym-Hybrid's original ``gym.spaces.Tuple`` action space
        to ``gym.spaces.Dict``.

   If the above wrapper does not meet your needs, you can also customize the wrapper yourself.

   It is worth mentioning that each wrapper also implements a static method ``new_shape``. The input parameters are the shape of observation, action, and reward before using the wrapper, and the output is the shape of the three after using the wrapper. This method will be used in the next section ``info``.

   It is worth mentioning that each wrapper must not only change of the corresponding observation/action/reward value, but also modify its corresponding space attribute accordingly (if and only when shpae, dtype, etc. are modified). And it will be discussed next section.


2. 3 Space Attributes: observation/action/reward space

   If you want to automatically create a neural network according to the dimensions of the environment, or use the ``shared_memory`` feature in ``EnvManager`` to speed up the transmission of environment's large tensor data, you need to provide property APIs: ``observation_space`` ``action_space`` ``reward_space``, in your env.

   For example, there are 3 properties that cartpole provides:

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

   Since cartpole does not use any wrappers, ``BaseEnvInfo`` is easire to specify. However, if an environment like Atari is decorated with multiple wrappers, you need to know what changes each wrapper has made to ``BaseEnvInfo``. That is why we must implement ``new_shape`` method in each wrapper in the previous section. Its usage is as follows:

   Since cartpole does not use any wrappers, its three spaces are fixed. But if the environment is decorated with multiple wrappers like Atari, it is necessary to modify the corresponding space after each wrapper decorates the original environment. For example, Atari will use ``ScaledFloatFrameWrapper`` to normalize observations to the interval [0, 1], and then modify its ``observation_space`` accordingly:

   .. code:: python

      class ScaledFloatFrameWrapper(gym.ObservationWrapper):
         
         def __init__(self, env):
            # ...
            self.observation_space = gym.spaces.Box(low=0., high=1., shape=env.observation_space.shape, dtype=np.float32)

3. ``enable_save_replay()``

   ``DI-engine`` does not mandate the implementation of the ``render`` method. If you want visualization, we recommend implementing ``enable_save_replay`` method.
   
   This method is called before the ``reset`` method and after the ``seed`` method. This method specifies the storage path of the video. It should be noted that this method does **not directly store the video**, but only sets a flag whether to save the video or not. The code of actually storing the video needs to be implemented by yourself. (Since multiple environments may be run at a time, and each environment runs multiple episodes, we recommend using episode_id and env_id in the file name to distinguish them)

   Here, an example in DI-engine is given, which uses the decorator provided by ``gym`` to encapsulate the environment, as shown in the code:

   .. code:: python

      class AtariEnv(BaseEnv):

         def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
            if replay_path is None:
               replay_path ='./video'
            self._replay_path = replay_path
            # this function can lead to the meaningless result
            # disable_gym_view_window()
            self._env = gym.wrappers.Monitor(
               self._env, self._replay_path, video_callable=lambda episode_id: True, force=True
            )

4. Use different configs for training environment and evaluation environment

   The environment used for training (collector_env) and the environment used for evaluation (evaluator_env) may use different configurations. A static method can be implemented in the environment to implement custom configuration for different environments' configuration. Take Atari as an example:

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

   The original configuration item ``cfg`` can be converted:

   .. code:: python

      # env_fn is an env class
      collector_env_cfg = env_fn.create_collector_env_cfg(cfg)
      evaluator_env_cfg = env_fn.create_evaluator_env_cfg(cfg)

   Setting the item ``cfg.is_train`` will use different decoration methods in the wrapper accordingly. For example, if ``cfg.is_train == True``, reward will be applied a sign function to map to ``{+1, 0, -1}`` for better training, if ``cfg.is_train == False`` The original reward value will be retained to facilitate the evaluation of the agent's performance during testing.

5. ``random_action()``

   Some off-policy algorithms require that before training starts, we can collect some data to insert into the buffer with a random strategy for initialization. Due to this requirement, DI-engine encourages to implement the ``random_action`` method.

   Since the environment already supports ``action_space`` property, you can directly call the ``Space.sample()`` method provided by gym to randomly select an action. But it should be noted that, since DI-engine requires all returned actions to be in ``np.ndarray`` format, some necessary transformations may be required. E.g:

   .. python::

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

DingEnvWrapper
~~~~~~~~~~~~~~~~~~~~~~~
(in ``ding/envs/env/ding_env_wrapper.py``)

``DingEnvWrapper`` can quickly convert simple environments such as cartpole, pendulum, etc. into environments that conform to ``BaseEnv``. However, more complex environments are not supported for the time being.

TBD


Q & A
~~~~~~~~~~~~~~

1. How should the MARL environment be migrated?
   
   You can refer to `Competitive RL <../env_tutorial/competitive_rl_zh.html>`_

   - If the environment supports both single-agent, double-agent or even multi-agent, you should fully consider those different modes
   - In a multi-agent environment, action and observation would match the number of agents, but reward and done are not always match. You need to figure out the definition of reward
   - Pay attention to how the original environment requires action and observation to be combined (tuples, lists, dictionaries, stacked arrays...)


2. How should the environment of the mixed action space be migrated?
   
   You can refer to `Gym-Hybrid <../env_tutorial/gym_hybrid_zh.html>`_

   - Some discrete actions (Accelerate, Turn) in Gym-Hybrid need to be given corresponding 1-dimensional continuous parameters to represent acceleration and rotation angle. Similar environments need to well define the action space
