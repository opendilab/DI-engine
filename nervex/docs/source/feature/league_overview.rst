League Manager Overview
========================

概述：
    league这一概念原本来自于AlphaStar。在AlphaStar中存在一个league，league中有很多player，每个player持有一个具有
    特定参数的网络，不同player之间可以互相交战，不断升级，即用league中多种player之间组合丰富的交战，代替了简单的self-play。

    StarCraft这种1v1的游戏模式，我们称为 **Battle**。
    Battle中的league负责为每个player分配对手，决定将与谁交战，这是AlphaStar最终的训练效果如此之好的原因之一。

    但在一些其他类型的环境中，如Atari, Mujuco等，我们称为 **Solo**， 并不存在两个player交战的情况，而是单个player单纯地与环境互动。
    这种场景下，league的设计还是否可以保留呢？答案是可以的。

    我们设计的league，将核心任务从的AlphaStar的分配对手，扩展到了分配工作：决定当前这个player是像actor一样去和环境交互，
    产生训练数据，还是像learner一样去从actor产生的训练数据中学习网络。

        - 如果是actor，那么交互时要开启多少个环境，每个环境的episode设置为多少，是否使用探索策略；
        - 如果是learner，是否要对训练数据进行处理，是否使用某种trick等等。

    而针对类似AlphaStar这种架构，便在众多信息的基础上，再添加一个分配的对手。

    我们的league manager分为三个部分：

        - player：league中的player，分为active（参数可更新）和historical（参数不可更新）两类，是分配工作、执行工作的单位。
        - payoff：用于记录league中player的以往记录，通常为所有player所共享，用于未来工作的分配。
        - league manager：持有全部player及他们的payoff记录，负责为根据payoff为每个player分配工作。


Player
---------

概述：
    player是league的成员，是分配工作、执行工作的单位。每个player会拥有一份网络参数，代表它会执行某种不完全和他人相同的策略。

    大体上，player可以分为active和historical两类：

        - active表示player的网络参数可以更新，通常是league中最重要的需要被训练的player；
        - historical表示它是某个时刻的active参数冻结产生的player，它在之后参数都将保持固定，常作为active player的对手，或者由于可以探索到环境中某个难以被探索的部分而作为actor冻结下来。

代码结构：
    主要实现如下几个类：

        1. Player: 基类，持有多个属性。
        2. ActivePlayer: 可更新的player，可以获取工作，在训练一段时间后拍快照产生historical_player以及变异。
        3. HistoricalPlayer: 不可更新的player。
        4. SoloActivePlayer: solo环境中的active_player，获取工作时无需寻找对手。
        5. BattleActivePlayer: battle环境中的active_player，获取工作时需要寻找对手。
        6. MainPlayer: 用于AlphaStar，battle环境中active_player的一种实现。
        7. MainExploiter: 用于AlphaStar，battle环境中active_player的一种实现。
        8. LeagueExploiter: 用于AlphaStar，battle环境中active_player的一种实现。

基类定义：
    1. Player (nervex/league/player.py)

        .. code:: python

            class Player:

                def __init__(self, cfg: EasyDict, category: str,
                            init_payoff: Union['BattleSharedPayoff', 'SoloSharedPayoff'],  # noqa
                            checkpoint_path: str, player_id: str, total_agent_step: int) -> None:
                    self._cfg = cfg
                    self._category = category
                    self._payoff = init_payoff
                    self._checkpoint_path = checkpoint_path
                    assert isinstance(player_id, str)
                    self._player_id = player_id
                    self._total_agent_step = total_agent_step

                @property
                def category(self) -> str:
                    return self._category

                @property
                def payoff(self) -> Union['BattleSharedPayoff', 'SoloSharedPayoff']:  # noqa
                    return self._payoff

                @property
                def checkpoint_path(self) -> str:
                    return self._checkpoint_path

                @property
                def player_id(self) -> str:
                    return self._player_id

                @property
                def total_agent_step(self) -> int:
                    return self._total_agent_step

                @total_agent_step.setter
                def total_agent_step(self, step: int) -> None:
                    self._total_agent_step = step

        - 概述：

            定义了active player和historical player都需要的一些属性。是抽象基类，故不能实例化。

    2. HistoricalPlayer (nervex/league/player.py)

        .. code:: python

            class HistoricalPlayer(Player):

                def __init__(self, *args, parent_id: str) -> None:
                    super(HistoricalPlayer, self).__init__(*args)
                    self._parent_id = parent_id

                @property
                def parent_id(self) -> str:
                    return self._parent_id

        - 概述：

            额外定义了parent id。historical player通常由active player ``snapshot`` 而来，或是在league初始化时主动
            添加预训练模型。

    3. ActivePlayer (nervex/league/player.py)

        .. code:: python

            class ActivePlayer(Player):

                def __init__(self, *args, **kwargs) -> None:
                    super(ActivePlayer, self).__init__(*args)
                    self._one_phase_step = int(float(self._cfg.one_phase_step))
                    self._last_enough_step = 0
                    if 'exploration' in self._cfg.forward_kwargs:
                        self._exploration = epsilon_greedy(self._cfg.forward_kwargs.exploration.start,
                                                        self._cfg.forward_kwargs.exploration.end,
                                                        self._cfg.forward_kwargs.exploration.decay_len)
                    else:
                        self._exploration = None

                def is_trained_enough(self, *args, **kwargs) -> bool:
                    step_passed = self._total_agent_step - self._last_enough_step
                    if step_passed < self._one_phase_step:
                        return False
                    else:
                        self._last_enough_step = self._total_agent_step
                        return True

                def snapshot(self) -> HistoricalPlayer:
                    path = self.checkpoint_path.split('.pth')[0] + '_{}'.format(self._total_agent_step) + '.pth'
                    return HistoricalPlayer(
                        self._cfg,
                        self.category,
                        self.payoff,
                        path,
                        self.player_id + '_{}'.format(int(self._total_agent_step)),
                        self._total_agent_step,
                        parent_id=self.player_id
                    )

                def mutate(self, info: dict) -> Optional[str]:
                    pass

                def get_job(self) -> dict:
                    job_dict = self._cfg.job
                    return deep_merge_dicts({
                        'forward_kwargs': self._get_job_forward(),
                        'adder_kwargs': self._get_job_adder(),
                        'env_kwargs': self._get_job_env()
                    }, job_dict)

                def _get_job_forward(self) -> dict:
                    ret = {}
                    if 'exploration' in self._cfg.forward_kwargs:
                        ret['eps'] = self._exploration(self.total_agent_step)
                    return ret

                def _get_job_adder(self) -> dict:
                    return self._cfg.adder_kwargs

                def _get_job_env(self) -> dict:
                    return self._cfg.env_kwargs

        - 概述：

            ``get_job`` 调用 ``_get_job_*`` 来获取工作，随后由league正式分配工作，player开始执行工作。执行一段时间后
            可以判断是否 ``is_trained_enough`` ，若是，则可以 ``snapshot`` 及 ``mutate`` ，然后继续获取新的工作。

        - 类接口方法：
            1. __init__: 初始化
            2. is_trained_enough: 是否得到了足够的训练，根据step数判定。
            3. snapshot: 冻结此时的网络参数，产生一个historical player并返回。
            4. mutate: 变异，比如可以对参数进行一些重置。
            5. get_job: 获取工作，调用 ``_get_job_*`` 几个子方法，返回一个包含job信息的dict


Payoff
----------------

概述：
    payoff用于记录以往的结果，该结果对于未来分配工作有着重要意义。
    例如，在battle环境中，胜率是选择对手时的考量指标之一，payoff便可以计算league中任意两个player间的胜率。

代码结构：
    主要分为如下几个子模块：

        1. BattleRecordDict: 继承自dict，表示battle环境中任意两个player间的对局情况。初始化时将四个key ['wins', 'draws', 'losses', 'games']的value置为0，支持乘法（用于decay）。
        2. PayoffDict: 继承自defaultdict，使得key不存在时可以默认返回一个BattleRecordDict(battle环境)或deque(solo环境)的实例。
        3. BattleSharedPayoff: 用于battle环境，可更新对战结果，计算胜率。
        4. SoloSharedPayoff: 用于solo环境，可更新对战结果。


League Manager
----------------

概述：
    league manager是管理player及他们之间关系（使用payoff），可统筹为player分配工作的类。
    是一个向量化的环境管理器，其中同时运行多个相同类型不同配置的环境，实际实现方式包含子进程向量化和伪向量化（循环串行）两种模式

基类定义：
    1. BaseLeagueManager (nervex/league/base_league_manager.py)

        .. code:: python


            class BaseLeagueManager(ABC):

                def __init__(self, cfg: EasyDict, save_checkpoint_fn: Callable, load_checkpoint_fn: Callable,
                        launch_job_fn: Callable) -> None:
                    self._init_cfg(cfg)
                    self.league_uid = str(uuid.uuid1())
                    self.active_players = []
                    self.historical_players = []
                    self.payoff = create_payoff(self.cfg.payoff)  # now supports ['solo', 'battle']
                    self.max_active_player_job = self.cfg.max_active_player_job
                    self.save_checkpoint_fn = save_checkpoint_fn
                    self.load_checkpoint_fn = load_checkpoint_fn
                    self.launch_job_fn = launch_job_fn
                    self._active_players_lock = LockContext(type_=LockContextType.THREAD_LOCK)
                    self._launch_job_thread = Thread(target=self._launch_job)
                    self._snapshot_thread = Thread(target=self._snapshot)
                    self._end_flag = False
                    self._init_league()

                def _init_cfg(self, cfg: EasyDict) -> None:
                    cfg = deep_merge_dicts(default_config, cfg)
                    self.cfg = cfg.league
                    self.model_config = cfg.model

                def _init_league(self) -> None:
                    for cate in self.cfg.player_category:
                        for k, n in self.cfg.active_players.items():
                            for i in range(n):
                                name = '{}_{}_{}_{}'.format(k, cate, i, self.league_uid)
                                ckpt_path = '{}_ckpt.pth'.format(name)
                                player = create_player(self.cfg, k, self.cfg[k], cate, self.payoff, ckpt_path, name, 0)
                                self.active_players.append(player)
                                self.payoff.add_player(player)

                    if self.cfg.use_pretrain_init_historical:
                        for cate in self.cfg.player_category:
                            name = '{}_{}_0_pretrain'.format('main_player', cate)
                            parent_name = '{}_{}_0'.format('main_player', cate)
                            hp = HistoricalPlayer(self.cfg.main_player, cate, self.payoff, self.cfg.pretrain_checkpoint_path[cate],
                                                name, 0, parent_id=parent_name)
                            self.historical_players.append(hp)
                            self.payoff.add_player(hp)

                    # register launch_count attribute for each active player
                    for p in self.active_players:
                        setattr(p, 'launch_count', LimitedSpaceContainer(0, self.max_active_player_job))

                    # save active players' player_id & player_ckpt
                    self.active_players_ids = [p.player_id for p in self.active_players]
                    self.active_players_ckpts = [p.checkpoint_path for p in self.active_players]
                    # validate active players are unique by player_id
                    assert len(self.active_players_ids) == len(set(self.active_players_ids))

                def finish_job(self, job_info: dict) -> None:
                    # update launch_count
                    with self._active_players_lock:
                        launch_player_id = job_info['launch_player']
                        idx = self.active_players_ids.index(launch_player_id)
                        self.active_players[idx].launch_count.release_space()
                    # save job info, update in payoff
                    self.payoff.update(job_info)

                def run(self) -> None:
                    self._launch_job_thread.start()
                    self._snapshot_thread.start()

                def close(self) -> None:
                    self._end_flag = True

                def _launch_job(self) -> None:
                    while not self._end_flag:
                        with self._active_players_lock:
                            # check whether there are empty job launchers in any player
                            launch_counts = [p.launch_count.get_residual_space() for p in self.active_players]
                            # launch job
                            if sum(launch_counts) != 0:
                                for idx, c in enumerate(launch_counts):
                                    for _ in range(c):
                                        player = self.active_players[idx]
                                        job_info = self._get_job_info(player)
                                        assert 'launch_player' in job_info.keys() and \
                                            job_info['launch_player'] == player.player_id
                                        self.launch_job_fn(job_info)
                        time.sleep(self.cfg.time_interval)

                @abstractmethod
                def _get_job_info(self, player: ActivePlayer) -> dict:
                    raise NotImplementedError

                def _snapshot(self) -> None:
                    time.sleep(int(0.5 * self.cfg.time_interval))
                    while not self._end_flag:
                        with self._active_players_lock:
                            # check whether there is an active player which is trained enough
                            flags = [p.is_trained_enough() for p in self.active_players]
                            if sum(flags) != 0:
                                for idx, f in enumerate(flags):
                                    if f:
                                        player = self.active_players[idx]
                                        # snapshot
                                        hp = player.snapshot()
                                        self.save_checkpoint_fn(player.checkpoint_path, hp.checkpoint_path)
                                        self.historical_players.append(hp)
                                        self.payoff.add_player(hp)
                                        # mutate
                                        self._mutate_player(player)
                        time.sleep(self.cfg.time_interval)

                @abstractmethod
                def _mutate_player(self, player: ActivePlayer) -> None:
                    raise NotImplementedError

                def update_active_player(self, player_info: dict) -> None:
                    try:
                        idx = self.active_players_ids.index(player_info['player_id'])
                        player = self.active_players[idx]
                        self._update_player(player, player_info)
                    except ValueError:
                        pass

                @abstractmethod
                def _update_player(self, player: ActivePlayer, player_info: dict) -> None:
                    raise NotImplementedError


        - 概述：

            league中会开启两个线程，一个是 ``_launch_job_thread`` ，会检查所有player是否还有分配工作的余量，若有则分配；
            另一个是 ``_snapshot_thread`` ，会检查所有player是否得到了充足的训练，若是则冻结参数产生historical player及变异。
            在job结束时，``finish_job`` 和 ``update_active_player`` 会被调用以更新信息。

        - 类接口方法：
            1. __init__: 初始化，在最前会调用 ``_init_cfg``，读取当前league manager的config；最后会调用 ``_init_league`` ，初始化league中的player。
            2. finish_job: 为某个结束工作的player释放job空间，并更新shared payoff
            3. run: 开启上述两个线程
            4. close: 将end flag设为True，使两个线程不再工作
            5. update_active_player: 在一个job结束后，更新active player的信息，如step

        .. note::

            _get_job_info(被_launch_job调用)，_mutate_player(被_snapshot调用)，
            _update_player(被update_active_player调用)三个方法没有实现，
            具体的实现方法可以参考测试文件 nervex/league/tests/test_league_manager.py中的 ``FakeLeagueManager``

