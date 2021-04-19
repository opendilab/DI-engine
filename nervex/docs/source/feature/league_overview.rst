League Overview
========================

概述：
    League这一概念原本来自于`AlphaStar <../rl_warmup/algorithm/large-scale-rl.html#alphastar>`_ 。
    在 AlphaStar 中存在一个 league，league 中有很多 player，每个 player 持有一个策略（或是网络），
    不同 player 之间可以互相交战，并根据对战数据更新自身策略。
    StarCraft2 这种 1v1 对战游戏中，league 负责为 player 分配对手，决定将与谁交战，这是 AlphaStar 训练效果非常好的原因之一。

    在nerveX中，也实现了 ``league`` 模块，可分为三个部分：

        - ``Player``：league 中的 player ，分为 active（策略可更新）和 historical（策略不可更新）两类。
        - ``Payoff``：用于记录 league 中全部 player 的全部对局结果，通常为所有 player 所共享，是未来分配对手的依据。
        - ``League``：持有全部 player 及他们的 payoff 记录，负责维护 payoff 及 player 状态，并根据 payoff 为 player 分配对手。


Player
---------

概述：
    player 是 league 的成员，是对局中的玩家。每个 player 会拥有一份网络参数，代表它会执行某种不完全和他人相同的策略。

    大体上，player 可以分为 active 和 historical 两类：

        - active 表示 player 的网络参数可以更新，通常是 league 中需要被训练的 player；
        - historical 表示它是某个时刻的 active 参数冻结产生的 player，它在之后参数都将保持固定，常作为 active player 的对手，它的存在提升了对局数据的多样性。

代码结构：
    主要实现如下几个类：

        1. ``Player``: player 基类，持有多个属性。
        2. ``ActivePlayer``: 可更新的 player，可以被分配对手并进行对局，在训练一段时间后可以通过 snapshot 产生 historical player。
        3. ``HistoricalPlayer``: 不可更新的player，通常可由以下两种方式得到——1）在训练最开始时加载一个预训练模型；2）active player 训练一段时间后 snapshot。
        4. ``NaiveSpPlayer``: active player的一种实现，可与 historical player 对战，或者进行简单的 self-play。
        5. ``MainPlayer``: active player 的一种实现，用于 AlphaStar，具体介绍可参考原论文。
        6. ``MainExploiter``: active player 的一种实现，用于 AlphaStar，具体介绍可参考原论文。
        7. ``LeagueExploiter``: active player 的一种实现，用于 AlphaStar，具体介绍可参考原论文。

基类定义：
    1. Player (nervex/league/player.py)

        .. code:: python

            class Player:

                def __init__(self, cfg: EasyDict, category: str,
                            init_payoff: Union['BattleSharedPayoff'],  # noqa
                            checkpoint_path: str, player_id: str, total_agent_step: int) -> None:
                    self._cfg = cfg
                    self._category = category
                    self._payoff = init_payoff
                    self._checkpoint_path = checkpoint_path
                    assert isinstance(player_id, str)
                    self._player_id = player_id
                    self._total_agent_step = total_agent_step

        - 概述：

            定义了 active player 和 historical player 都需要的一些属性，如类别、共享 payoff、checkpoint路径、id、持有网络的训练步数等。
            是抽象基类，故不能实例化。

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

            额外定义了 parent id。

    3. ActivePlayer (nervex/league/player.py)

        .. code:: python

            class ActivePlayer(Player):
                _name = "ActivePlayer"
                BRANCH = namedtuple("BRANCH", ['name', 'prob'])

                def __init__(self, *args, **kwargs) -> None:
                    """
                    Overview:
                        Initialize player metadata, depending on the game
                    Note:
                        - one_phase_step (:obj:`int`): An active player will be considered trained enough for snapshot \
                            after two phase steps.
                        - last_enough_step (:obj:`int`): Player's last step number that satisfies ``_is_trained_enough``.
                        - strong_win_rate (:obj:`float`): If win rates between this player and all the opponents are greater than
                            this value, this player can be regarded as strong enough to these opponents. \
                            If also already trained for one phase step, this player can be regarded as trained enough for snapshot.
                        - branch_probs (:obj:`namedtuple`): A namedtuple of probabilities of selecting different opponent branch.
                    """
                    # ...

                def is_trained_enough(self, select_fn: Optional[Callable] = None) -> bool:
                    """
                    Overview:
                        Judge whether this player is trained enough for further operations(e.g. snapshot, mutate...)
                        according to past step count and overall win rates against opponents.
                        If yes, set ``self._last_agent_step`` to ``self._total_agent_step`` and return True; otherwise return False.
                    Arguments:
                        - select_fn (:obj:`function`): The function to select opponent players.
                    Returns:
                        - flag (:obj:`bool`): Whether this player is trained enough
                    """
                    # ...

                def snapshot(self) -> HistoricalPlayer:
                    """
                    Overview:
                        Generate a snapshot historical player from the current player, called in league's ``_snapshot``.
                    Returns:
                        - snapshot_player (:obj:`HistoricalPlayer`): new instantiated historical player

                    .. note::
                        This method only generates a historical player object, but without saving the checkpoint, which should be
                        done by league.
                    """
                    # ...

                def mutate(self, info: dict) -> Optional[str]:
                    """
                    Overview:
                        Mutate the current player, called in league's ``_mutate_player``.
                    Arguments:
                        - info (:obj:`dict`): related information for the mutation
                    Returns:
                        - mutation_result (:obj:`str`): if the player does the mutation operation then returns the
                            corresponding model path, otherwise returns None
                    """
                    # ...

                def get_job(self, eval_flag: bool = False) -> dict:
                    """
                    Overview:
                        Get a dict containing some info about the job to be launched, e.g. the selected opponent.
                    Arguments:
                        - eval_flag (:obj:`bool`): Whether to select an opponent for evaluator task.
                    Returns:
                        - ret (:obj:`dict`): The returned dict. Should contain key ['opponent'].
                    """
                    # ...

                def _get_collect_opponent(self) -> Player:
                    """
                    Overview:
                        Select an opponent according to the player's ``branch_probs``.
                    Returns:
                        - opponent (:obj:`Player`): Selected opponent.
                    """
                    # ...

                def _get_players(self, select_fn: Callable) -> List[Player]:
                    """
                    Overview:
                        Get a list of players in the league (shared_payoff), selected by ``select_fn`` .
                    Arguments:
                        - select_fn (:obj:`function`): players in the returned list must satisfy this function
                    Returns:
                        - players (:obj:`list`): a list of players that satisfies ``select_fn``
                    """
                    # ...

                def _get_opponent(self, players: list, p: Optional[np.ndarray] = None) -> Player:
                    """
                    Overview:
                        Get one opponent player from list ``players`` according to probability ``p``.
                    Arguments:
                        - players (:obj:`list`): a list of players that can select opponent from
                        - p (:obj:`np.ndarray`): the selection probability of each player, should have the same size as \
                            ``players``. If you don't need it and set None, it would select uniformly by default.
                    Returns:
                        - opponent_player (:obj:`Player`): a random chosen opponent player according to probability
                    """
                    # ...

                def increment_eval_difficulty(self) -> bool:
                    """
                    Overview:
                        When evaluating, active player will choose a specific builtin opponent difficulty.
                        This method is used to increment the difficulty.
                        It is usually called after the easier builtin bot is already been beaten by this player.
                    Returns:
                        - increment_or_not (:obj:`bool`): True means difficulty is incremented; \
                            False means difficulty is already the hardest.
                    """
                    # ...

        - 概述：

            league 在被 commander 调用需要生成新的 collector job 时，将调用指定 player 的 ``get_job`` 方法，获取其对手。
            在 collector 开始执行任务后，learner 利用产生的数据训练自身，训练一段时间后，会通过 commander 告知 league，
            然后 league 调用指定 player 的 ``is_trained_enough`` 方法，判断当前产生数据的 collector 所持有的策略是否相对更新了较多：
            若是，则可以 ``snapshot`` 及 ``mutate``。

        - 类接口方法：
            1. ``__init__``: 初始化
            2. ``is_trained_enough``: 是否得到了足够的训练，根据step数判定。
            3. ``snapshot``: 冻结此时的网络参数，产生一个historical player并返回。
            4. ``mutate``: 变异，比如可以对参数进行一些重置。
            5. ``get_job``: 获取任务，调用 ``_get_job_opponent`` 获取对手。
        
        - 需要用户实现的方法：

            ``ActivePlayer`` 中没有实现具体的寻找对手的方法。寻找对手的逻辑为：首先一类 active player 应当会根据一种或多种不同的策略选择对手的类别，比如 ``NaiveSpPlayer`` 有 50% 的概率进行简单的 self-play，还有 50% 的概率从所有 Historical player 中任选一个。
            
            为了实现该过程，需要在 config 和类方法两处进行对应实现。下面依然以 ``NaiveSpPlayer`` 为例。
            
            1. config

                .. code:: python

                    # in nervex/config/league.py
                    naive_sp_player=dict(
                        # ...
                        branch_probs=dict(
                            pfsp=0.5,
                            sp=0.5,
                        ),
                    )
                
            2. ``NaiveSpPlayer`` 中实现的两个方法

                .. code:: python
                    
                    class NaiveSpPlayer(ActivePlayer):
                        
                        def _pfsp_branch(self) -> HistoricalPlayer:
                        """
                        Overview:
                            Select prioritized fictitious self-play opponent, should be a historical player.
                        Returns:
                            - player (:obj:`HistoricalPlayer`): The selected historical player.
                        """
                        # ...
                        return self._get_opponent(historical, p)

                    def _sp_branch(self) -> ActivePlayer:
                        """
                        Overview:
                            Select normal self-play opponent
                        """
                        return self


Payoff
----------------

概述：
    payoff 用于记录以往对局的结果，该结果对于未来分配对手有着重要意义。
    例如，在对战的环境中，胜率是选择对手时的考量指标之一，payoff 便可以计算 league 中任意两个 player 间的胜率。

代码结构：
    主要分为如下两个子模块：

        1. ``BattleRecordDict``: 继承自 dict，记录两个 player 间的对局情况。初始化时将四个 key ['wins', 'draws', 'losses', 'games']的 value 置为0。
        2. ``BattleSharedPayoff``: 利用 ``BattleRecordDict``，可记录 league 中任意两个 player 之间的对战结果，并计算胜率。


League
----------------

概述：
    league 是管理 player 及他们之间关系 （使用payoff），可统筹为 player 分配工作的类。
    一般由 Commander 持有一个，用于在对战类环境中生成 collector task 中，找到合适的两个 player 参与该对局。

基类定义：
    1. BaseLeague (nervex/league/base_league.py)

        .. code:: python

            class BaseLeague(ABC):
                """
                Overview:
                    League, proposed by Google Deepmind AlphaStar. Can manage multiple players in one league.
                Interface:
                    __init__, get_job_info, judge_snapshot, update_active_player, finish_job
                """

                def __init__(self, cfg: EasyDict) -> None:
                    """
                    Overview:
                        Initialization method.
                    Arguments:
                        - cfg (:obj:`EasyDict`): League config.
                    """
                    self._init_cfg(cfg)
                    # ...
                    self._init_players()

                @abstractmethod
                def _init_cfg(self, cfg: EasyDict) -> None:
                    """
                    Overview:
                        Initialize config ``self.cfg``.
                    """
                    raise NotImplementedError

                def _init_players(self) -> None:
                    """
                    Overview:
                        Initialize players (active & historical) in the league.
                    """
                    # Add different types of active players for each player category, according to ``cfg.active_players``.
                    # ...
                    # Add pretrain player as the initial HistoricalPlayer for each player category.
                    # ...

                def get_job_info(self, player_id: str = None, eval_flag: bool = False) -> dict:
                    """
                    Overview:
                        Get info of the job which is to be launched to an active player.
                    Arguments:
                        - player_id (:obj:`str`): The active player's id.
                        - eval_flag (:obj:`bool`): Whether this is an evaluation job.
                    Returns:
                        - job_info (:obj:`dict`): Job info. Should include keys ['lauch_player'].
                    """
                    # ...

                @abstractmethod
                def _get_job_info(self, player: ActivePlayer, eval_flag: bool = False) -> dict:
                    """
                    Overview:
                        Real get_job method. Called by ``_launch_job``.
                    Arguments:
                        - player (:obj:`ActivePlayer`): The active player to be launched a job.
                        - eval_flag (:obj:`bool`): Whether this is an evaluation job.
                    Returns:
                        - job_info (:obj:`dict`): Job info. Should include keys ['lauch_player'].
                    """
                    raise NotImplementedError

                def judge_snapshot(self, player_id: str) -> bool:
                    """
                    Overview:
                        Judge whether a player is trained enough for snapshot. If yes, call player's ``snapshot``, create a
                        historical player(prepare the checkpoint and add it to the shared payoff), then mutate it, and return True.
                        Otherwise, return False. 
                    Arguments:
                        - player_id (:obj:`ActivePlayer`): The active player's id.
                    Returns:
                        - snapshot_or_not (:obj:`dict`): Whether the active player is snapshotted.
                    """
                    # ...

                @abstractmethod
                def _mutate_player(self, player: ActivePlayer) -> None:
                    """
                    Overview:
                        Players have the probability to mutate, e.g. Reset network parameters.
                        Called by ``self._snapshot``.
                    Arguments:
                        - player (:obj:`ActivePlayer`): The active player that may mutate.
                    """
                    raise NotImplementedError

                def update_active_player(self, player_info: dict) -> None:
                    """
                    Overview:
                        Update an active player's info.
                    Arguments:
                        - player_info (:obj:`dict`): Info dict of the player which is to be updated, \
                            at least includs ['player_id', 'train_iteration']
                    """
                    # ...

                @abstractmethod
                def _update_player(self, player: ActivePlayer, player_info: dict) -> None:
                    """
                    Overview:
                        Update an active player. Called by ``self.update_active_player``.
                    Arguments:
                        - player (:obj:`ActivePlayer`): The active player that will be updated.
                        - player_info (:obj:`dict`): Info dict of the active player which is to be updated.
                    """
                    raise NotImplementedError

                def finish_job(self, job_info: dict) -> None:
                    """
                    Overview:
                        Finish current job. Update shared payoff to record the game results.
                    Arguments:
                        - job_info (:obj:`dict`): A dict containing job result information.
                    """
                    # ...

                @staticmethod
                def save_checkpoint(src_checkpoint, dst_checkpoint) -> None:
                    '''
                    Overview:
                        Copy a checkpoint from path ``src_checkpoint`` to path ``dst_checkpoint``.
                    Arguments:
                        - src_checkpoint (:obj:`str`): Source checkpoint's path, e.g. s3://alphastar_fake_data/ckpt.pth
                        - dst_checkpoint (:obj:`str`): Destination checkpoint's path, e.g. s3://alphastar_fake_data/ckpt.pth
                    '''
                    # ...


        - 概述：

            league 完全接受 commander 的命令，为 commander 在对战环境中提供对战双方的信息。

        - 类接口方法：
            1. ``__init__``: 初始化，在最前会调用 ``_init_cfg``，读取当前league的config；最后会调用 ``_init_league`` ，初始化league中的player。
            2. ``get_job_info``: commander 在准备为 collector 分配任务后，调用该方法得知此次任务由哪两个 player 执行。
            3. ``judge_snapshot``: 当 learner 利用数据更新自身策略后，player 持有的策略也会相应更新，在一定时间的训练后，commander 调用此方法判断 player 的策略是否得到了足够的训练。
            4. ``update_active_player``: 当 learner 训练后，或是 evaluator 结束评估后，更新对应 player 的模型步数或下一次 evaluate 中将选择的对手。
            5. ``finish_job``: 当 collector 任务结束后，更新 shared payoff 中的对战信息。

        - 需要用户实现的方法：

            ``_get_job_info`` (被 ``_launch_job`` 调用)，``_mutate_player`` (被 ``_snapshot`` 调用)，
            ``_update_player`` (被 ``update_active_player`` 调用)三个方法均为抽象方法，
            具体的实现方法可以参考 ``nervex/league/one_vs_one_league.py`` 中的 ``OneVsOneLeague``

