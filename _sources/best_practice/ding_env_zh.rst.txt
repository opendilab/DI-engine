如何将自己的环境迁移到DI-engine中
==============================================================

虽然已经在 ``DI-zoo`` 中提供了大量的强化学习常用环境（ `已支持的环境 <https://github.com/opendilab/DI-engine#environment-versatility>`_ ），但用户还是会需要将自己的环境迁移到 ``DI-engine`` 中。因此在本节中，将会介绍如何一步步进行上述迁移，以满足 ``DI-engine`` 的基础环境基类 ``BaseEnv`` 的规范，从而轻松应用在训练的 pipeline 中。

下面的介绍将分为 **基础** 和 **进阶** 两部分。 **基础** 表明如果想跑通 pipeline 必须实现的功能和注意的细节； **进阶** 则表示一些拓展的功能。

基础
~~~~~~~~~~~~~~

本节将介绍迁移环境时，用户必须满足的规范约束、以及必须实现的功能。

如果要在 DI-engine 中使用环境，需要实现一个继承自 ``BaseEnv`` 的子类环境，例如 ``YourEnv`` 。 ``YourEnv`` 和你自己的环境之间是 `组合 <https://www.cnblogs.com/chinxi/p/7349768.html>`_ 关系，即在一个 ``YourEnv`` 实例中，会持有一个你自己的环境的实例。

强化学习的环境有一些普遍地、被大多数环境实现了的主要接口，如 ``reset()``, ``step()``, ``seed()`` 等。在 DI-engine 中， ``BaseEnv`` 将对这些接口进行进一步的封装，下面大部分情况下将以 Atari 为例进行说明。具体代码可以参考 `Atari Env <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/envs/atari_env.py>`_ 和 `Atari Env Wrapper <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/envs/atari_wrappers.py>`_


1. ``__init__()``

   一般情况下，可能会在 ``__init__`` 方法中将环境实例化，但是在 DI-engine 中，为了便于支持 ``EnvManager`` 中的“环境向量化”等并行操作，环境实例一般采用 **Lazy Init** 的方式，即 ``__init__`` 方法不初始化真正的原始环境实例，只是设置相关 **参数配置值** ，在第一次调用 ``reset`` 方法时，才会进行实际的环境初始化。

   以 Atari 为例。 ``__init__`` 并不实例化环境，只是设置配置项 ``self._cfg`` ，以及初始化变量 ``self._init_flag`` 用于记录是否是第一次调用 ``reset`` 方法（即环境是否还没有被初始化）。


   .. code:: python
      
      class AtariEnv(BaseEnv):

         def __init__(self, cfg: dict) -> None:
            self._cfg = cfg
            self._init_flag = False

2. ``seed()``

   ``seed`` 用于设定环境中的随机种子，环境中有两部分随机种子需要设置，一是 **原始环境** 的随机种子，二是各种 **环境变换** 中调用随机库时的随机种子（例如 ``random``， ``np.random``）。如果你的环境完全没有任何随机性（包括“原始环境”与“环境变换”），那么也可以不实现这个方法。

   随机库的种子的设置较为简单，直接在环境的 ``seed`` 方法中进行设置。

   但原始环境的种子，在 ``seed`` 方法中只是进行了赋值，并没有真的设置；真正的设置是在调用环境的 ``reset`` 方法内部，具体的原始环境 ``reset`` 之前进行设置。

   .. code:: python

      class AtariEnv(BaseEnv):
         
         def seed(self, seed: int, dynamic_seed: bool = True) -> None:
            self._seed = seed
            self._dynamic_seed = dynamic_seed
            np.random.seed(self._seed)

   针对原始环境的种子，DI-engine 中有 **静态种子** 和 **动态种子** 的概念。
   
   **静态种子** 用于测试环境，保证每个 episode 的随机种子相同，即 ``reset`` 时只会采用 ``self._seed`` 这个固定的静态种子数值。需要在 ``seed`` 方法中手动传入 ``dynamic_seed`` 参数为 ``False`` 。

   **动态种子** 用于训练环境，尽量使得每个 episode 的随机种子都不相同，它们都在 ``reset`` 方法中由一个随机数发生器 ``100 * np.random.randint(1, 1000)`` 产生（但这个随机数发生器的种子是通过环境的 ``seed`` 方法固定的，因此能保证实验的可复现性）。需要在 ``seed`` 方法中不传入 ``dynamic_seed`` 参数，或者传入参数为 ``True``。

3. ``reset()``

   在 ``__init__`` 方法中已经介绍了 DI-engine 的 **Lazy Init** 初始化方式，即实际的环境初始化是在 **第一次调用** ``reset`` 方法时进行的。

   ``reset`` 方法中会根据 ``self._init_flag`` 判断是否需要实例化实际环境，并进行随机种子的设置，然后调用原始环境的 ``reset`` 方法得到初始状态下的观测值，并转换为 ``np.ndarray`` 数据格式（将在 5. 中详细讲解），并初始化 ``self._final_eval_reward`` 的值（将在 4. 中详细讲解），在 Atari 中 ``self._final_eval_reward`` 指的是一整个 episode 所获得的真实 reward 的累积和，用于评价 agent 在该环境上的性能，不用于训练。

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

   ``step`` 方法负责接收当前时刻的 ``action`` ，然后给出当前时刻的 ``reward`` 和 下一时刻的 ``obs``，在 DI-engine中，还需要给出：当前episode是否结束的标志 ``done``、字典形式的其它信息 ``info`` （比如 ``self._final_eval_reward`` ）。

   在得到 ``reward`` ``obs`` ``done`` ``info`` 等数据后，需要进行处理，转化为 ``np.ndarray`` 格式，以保证符合 DI-engine 的规范。在每一个时间步中 ``self._final_eval_reward`` 都会累加当前的真实 reward，并在 episode 结束（ ``done == True`` ）的时候返回该累加值。

   最终，将上述四个数据放入定义为 ``namedtuple`` 的 ``BaseEnvTimestep`` 中并返回（定义为： ``BaseEnvTimestep = namedtuple('BaseEnvTimestep', ['obs', 'reward', 'done', 'info'])`` ）
   
   .. code:: python

      from ding.envs import BaseEnvTimestep

      class AtariEnv(BaseEnv):
         
         def step(self, action: np.ndarray) -> BaseEnvTimestep:
            assert isinstance(action, np.ndarray), type(action)
            action = action.item()
            obs, rew, done, info = self._env.step(action)
            self._final_eval_reward += rew
            obs = to_ndarray(obs)
            rew = to_ndarray([rew])  # Transformed to an array with shape (1,)
            if done:
               info['final_eval_reward'] = self._final_eval_reward
            return BaseEnvTimestep(obs, rew, done, info)

5. ``self._final_eval_reward``

   在 Atari 环境中， ``self._final_eval_reward`` 是指一个 episode 的全部 reward 的累加和。

      - 在 ``reset`` 方法中，将当前 ``self._final_eval_reward`` 置0；
      - 在 ``step`` 方法中，将每个时间步获得的 reward 加到 ``self._final_eval_reward`` 中。
      - 在 ``step`` 方法中，如果当前 episode 已经结束（ ``done == True`` 此处要求 ``done`` 必须是 ``bool`` 类型，不能是 ``np.bool`` ），那么就添加到 ``info`` 这个字典中并返回： ``info['final_eval_reward'] = self._final_eval_reward``

   但是，在其他的环境中，可能需要的不是一个 episode 的 reward 之和。例如，在 smac 中，需要当前 episode 的胜率，因此就需要修改第二步 ``step`` 方法中简单的累加，改为记录对局情况，并最终在 episode 结束时返回计算得到的胜率。

6. 数据规格

   DI-engine 中要求环境中每个方法的输入输出的数据必须为 ``np.ndarray`` 格式，数据类型dtype 需要是 ``np.int64`` (整数) 或 ``np.float32`` (浮点数)。包括：

      -  ``reset`` 方法返回的 ``obs``
      -  ``step`` 方法接收的 ``action``
      -  ``step`` 方法返回的 ``obs``
      -  ``step`` 方法返回的 ``reward``，此处还要求 ``reward`` 必须为 **一维** ，而不能是零维，例如 Atari 中的代码 ``rew = to_ndarray([rew])`` 
      -  ``step`` 方法返回的 ``done``，必须是 ``bool`` 类型，不能是 ``np.bool``


进阶
~~~~~~~~~~~~

1. 环境预处理wrapper

   很多环境如果要用于强化学习的训练中，都需要进行一些预处理，来达到增加随机性、数据归一化、易于训练等目的。这些预处理通过 wrapper 的形式实现（wrapper 的介绍可以参考 `这里 <../feature/wrapper_hook_overview_zh.html#wrapper>`_ ）。
   
   环境预处理的每个 wrapper 都是 ``gym.Wrapper`` 的一个子类。例如， ``NoopResetEnv`` 是在 episode 最开始时，执行随机数量的 No-Operation 动作，是增加随机性的一种手段，其使用方法是：
   
   .. code:: python
      
      env = gym.make('PongNoFrameskip-v4')
      env = NoopResetEnv(env)
   
   由于 ``NoopResetEnv`` 中实现了 ``reset`` 方法，因此在 ``env.reset()`` 时就会执行 ``NoopResetEnv`` 中的相应逻辑。

   DI-engine 中已经实现了以下 env wrapper：(in ``ding/envs/env_wrappers/env_wrappers.py``)

      - ``NoopResetEnv``: 在 episode 最开始时，执行随机数量的 No-Operation 动作
      - ``MaxAndSkipEnv``: 返回几帧中的最大值，可认为是时间步上的一种 max pooling
      - ``WarpFrame``: 将原始的图像画面利用 ``cv2`` 库的 ``cvtColor`` 转换颜色编码，并 resize 为一定长宽的图像（一般为 84x84）
      - ``ScaledFloatFrame``: 将 observation 归一化到 [0, 1] 区间内（保持 dtype 为 ``np.float32`` ）
      - ``ClipRewardEnv``: 将 reward 通过一个符号函数，变为 ``{+1, 0, -1}``
      - ``FrameStack``: 将一定数量（一般为4）的 frame 堆叠在一起，作为新的 observation，可被用于处理 POMDP 的情况，例如，单帧信息无法知道运动的速度方向
      - ``ObsTransposeWrapper``: 将 ``(H, W, C)`` 的图像转换为 ``(C, H, W)`` 的图像
      - ``ObsNormEnv``: 利用 ``RunningMeanStd`` 将 observation 进行滑动窗口归一化
      - ``RewardNormEnv``: 利用 ``RunningMeanStd`` 将 reward 进行滑动窗口归一化
      - ``RamWrapper``: 将 Ram 类型的环境的 observation 的 shape 转换为类似图像的 (128, 1, 1)
      - ``EpisodicLifeEnv``: 将内置多条生命的环境（例如Qbert），将每条生命看作一个 episode
      - ``FireResetEnv``: 在环境 reset 后立即执行动作1（开火）
      - ``GymHybridDictActionWrapper``: 将 Gym-Hybrid 环境原始的 ``gym.spaces.Tuple`` 类型的动作空间，转换为 ``gym.spaces.Dict`` 类型的动作空间.

   如果上述 wrapper 不能满足你的需要，也可以自行定制 wrapper。

   值得一提的是，每个 wrapper 不仅要完成对相应的 observation/action/reward 值的变化，还要对应地修改其 space （当且仅当 shpae, dtype 等被修改时），这个方法将在下一节中详细介绍。

2. 三个空间属性 observation/action/reward space

   如果希望可以根据环境的维度自动创建神经网络，或是在 ``EnvManager`` 中使用 ``shared_memory`` 技术加快环境返回的大型张量数据的传输速度，就需要让环境支持提供属性 ``observation_space`` ``action_space`` ``reward_space``。
   
   这里的 space 都是 ``gym.spaces.Space`` 的子类的实例，最常用的 ``gym.spaces.Space`` 包括 ``Discrete`` ``Box`` ``Tuple`` ``Dict`` 等。space 中需要给出 **shape** 和 **dtype** 。在 gym 原始环境中，大多都会支持 ``observation_space`` ``action_space`` 和 ``reward_range``，在 DI-engine 中，将 ``reward_range`` 也扩充成了 ``reward_space``，使这三者保持一致。

   例如，这个是 cartpole 的三个属性：

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

   由于 cartpole 没有使用任何 wrapper，因此其三个 space 是固定不变的。但如果像 Atari 这种经过了多重 wrapper 装饰的环境，就需要在每个 wrapper 对原始环境进行包装之后，修改其对应的 space。例如，Atari 会使用 ``ScaledFloatFrameWrapper``，将 observation 归一化到 [0, 1] 区间内，那么相应地，就会修改其 ``observation_space``：

   .. code:: python

      class ScaledFloatFrameWrapper(gym.ObservationWrapper):
         
         def __init__(self, env):
            # ...
            self.observation_space = gym.spaces.Box(low=0., high=1., shape=env.observation_space.shape, dtype=np.float32)


3. ``enable_save_replay()``

   ``DI-engine`` 并没有强制要求实现 ``render`` 方法，如果想完成可视化，我们推荐实现 ``enable_save_replay`` 方法，对游戏视频进行保存。
   
   该方法在 ``reset`` 方法之前， ``seed`` 方法之后被调用，在该方法中指定录像存储的路径。需要注意的是，该方法并 **不直接存储录像**，只是设置一个是否保存录像的 flag。真正存储录像的代码和逻辑需要自己实现。（由于可能会开启多个环境，每个环境运行多个 episode，因此我们建议在文件名中用 episode_id 和 env_id 进行区分）

   此处，给出 DI-engine 中的一个例子，该例子利用 ``gym`` 提供的装饰器封装环境，如代码所示：

   .. code:: python

      class AtariEnv(BaseEnv):

         def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
            if replay_path is None:
               replay_path = './video'
            self._replay_path = replay_path
            # this function can lead to the meaningless result
            # disable_gym_view_window()
            self._env = gym.wrappers.Monitor(
               self._env, self._replay_path, video_callable=lambda episode_id: True, force=True
            )

4. 训练环境和测试环境使用不同 config

   用于训练的环境（collector_env）和用于测试的环境（evaluator_env）可能使用不同的配置项，可以在环境中实现一个静态方法来实现对于不同环境配置项的自定义配置，以 Atari 为例：

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

   在实际使用时，可以对原始的配置项 ``cfg`` 进行转换：

   .. code:: python

      # env_fn is an env class
      collector_env_cfg = env_fn.create_collector_env_cfg(cfg)
      evaluator_env_cfg = env_fn.create_evaluator_env_cfg(cfg)

   设置 ``cfg.is_train`` 项，将相应地在 wrapper 中使用不同的修饰方式。例如，若 ``cfg.is_train == True`` ，则将对 reward 使用符号函数映射至 ``{+1, 0, -1}`` 方便训练，若 ``cfg.is_train == False`` 则将保留原 reward 值，方便测试时评估 agent 的性能。

5. ``random_action()``

   一些 off-policy 算法希望可以在训练开始之前，用随机策略收集一些数据填充 buffer，完成 buffer 的初始化。出于这样的需求，DI-engine 鼓励实现 ``random_action`` 方法。

   由于环境已经实现了 ``action_space``，所以可以直接调用 gym 中提供的 ``Space.sample()`` 方法来随机选取动作。但需要注意的是，由于 DI-engine 要求所有返回的 action 需要是 ``np.ndarray`` 格式的，所以可能需要做一些必要的格式转换。例如：

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
~~~~~~~~~~~~~~~~~~~~~~~~
(in ``ding/envs/env/ding_env_wrapper.py``)

``DingEnvWrapper`` 可以快速将 ClassicControl, Box2d, Atari, Mujoco, GymHybrid 等简单环境转换为符合 ``BaseEnv`` 的环境。

可以查看 `使用实例 <https://github.com/opendilab/DI-engine/blob/main/ding/envs/env/tests/test_ding_env_wrapper.py>`_ 获取更多信息。



Q & A
~~~~~~~~~~~~~~

1. MARL 环境应当如何迁移？
   
   可以参考 `Competitive RL <../env_tutorial/competitive_rl_zh.html>`_ 

   - 如果环境既支持 single-agent，又支持 double-agent 甚至 multi-agent，那么要针对不同的模式分类考虑
   - 在 multi-agent 环境中，action 和 observation 和 agent 个数匹配，但 reward 和 done 却不一定，需要搞清楚 reward 的定义
   - 注意原始环境要求 action 和 observation 怎样组合在一起（元组、列表、字典、stacked array...）


2. 混合动作空间的环境应当如何迁移？
   
   可以参考 `Gym-Hybrid <../env_tutorial/gym_hybrid_zh.html>`_

   - Gym-Hybrid 中部分离散动作（Accelerate，Turn）是需要给出对应的 1 维连续参数的，以表示加速度和旋转角度，因此类似的环境需要主要关注其动作空间的定义