import os.path as osp
import uuid
import time
from abc import ABC, abstractmethod
from threading import Thread
from typing import Callable
from easydict import EasyDict

from nervex.league.player import ActivePlayer, HistoricalPlayer
from nervex.league.player import create_player
from nervex.league.shared_payoff import create_payoff
from nervex.utils import deep_merge_dicts, LockContextType, LimitedSpaceContainer
from nervex.utils import read_config, LockContext, import_module


class BaseLeague(ABC):
    """
    Overview:
        League, proposed by Google Deepmind AlphaStar. Can manage multiple players in one league.
    Interface:
        __init__, run, close, finish_job, update_active_player
    """

    def __init__(
            self, cfg: EasyDict, save_checkpoint_fn: Callable, load_checkpoint_fn: Callable, launch_job_fn: Callable
    ) -> None:
        """
        Overview:
            Initialization method.
        Arguments:
            - cfg (:obj:`EasyDict`): League config.
            - save_checkpoint_fn (:obj:`function`): The function used to save ckpt.
            - load_checkpoint_fn (:obj:`function`): The function used to load ckpt.
            - launch_job_fn (:obj:`function`): The function used to launch job.
        Note:
            launch_job_fn:
                Arguments:
                    - job_info (:obj:`dict`)
            save_checkpoint_fn:
                Arguments:
                    - src_checkpoint (:obj:`str`): src must be an existing path
                    - dst_checkpoint (:obj:`str`)
            load_checkpoint_fn:
                Arguments:
                    - player_id: (:obj:`str`)
                    - checkpoint_path: (:obj:`str`)
            job_info (:obj:`dict`)
                - launch_player (:obj:`str`)
            player_info (:obj:`dict`)
                - player_id (:obj:`str`)
                - train_step (:obj:`int`)
        """
        self._init_cfg(cfg)
        self.league_uid = str(uuid.uuid1())
        self.active_players = []
        self.historical_players = []
        # Supports ['solo', 'battle']
        self.payoff = create_payoff(self.cfg.payoff)
        self.max_active_player_job = self.cfg.max_active_player_job

        self.save_checkpoint_fn = save_checkpoint_fn
        self.load_checkpoint_fn = load_checkpoint_fn
        self.launch_job_fn = launch_job_fn
        self._active_players_lock = LockContext(type_=LockContextType.THREAD_LOCK)
        # todo
        self._launch_job_thread = Thread(target=self._launch_job, name='league_launch_job')
        # todo
        self._snapshot_thread = Thread(target=self._snapshot, name='league_snapshot')
        self._end_flag = False

        self._init_league()

    def _init_cfg(self, cfg: EasyDict) -> None:
        """
        Overview:
            Initialize config. Diffenrent leagues (e.g. solo, battle) have different default conifgs.
        """
        raise NotImplementedError

    def _init_league(self) -> None:
        """
        Overview:
            Initialize players (active & historical) in the league.
        """
        # Add different types of active players for each player category, according to ``cfg.active_players``.
        for cate in self.cfg.player_category:  # Player's category (Depends on the env)
            for k, n in self.cfg.active_players.items():  # Active player's type (Different in solo and battle)
                for i in range(n):  # This type's active player number
                    name = '{}_{}_{}_{}'.format(k, cate, i, self.league_uid)
                    ckpt_path = '{}_ckpt.pth'.format(name)
                    player = create_player(self.cfg, k, self.cfg[k], cate, self.payoff, ckpt_path, name, 0)
                    if self.cfg.use_pretrain:
                        self.save_checkpoint_fn(self.cfg.pretrain_checkpoint_path[cate], player.checkpoint_path)
                    self.active_players.append(player)
                    self.payoff.add_player(player)

        # Add pretrain player as the initial HistoricalPlayer for each player category.
        if self.cfg.use_pretrain_init_historical:
            for cate in self.cfg.player_category:
                main_player_name = [k for k in self.cfg.keys() if 'main_player' in k]
                assert len(main_player_name) == 1, main_player_name
                main_player_name = main_player_name[0]
                name = '{}_{}_0_pretrain'.format(main_player_name, cate)
                parent_name = '{}_{}_0'.format(main_player_name, cate)
                hp = HistoricalPlayer(
                    self.cfg.get(main_player_name),
                    cate,
                    self.payoff,
                    self.cfg.pretrain_checkpoint_path[cate],
                    name,
                    0,
                    parent_id=parent_name
                )
                self.historical_players.append(hp)
                self.payoff.add_player(hp)

        # Register launch_count attribute for each active player.
        for p in self.active_players:
            setattr(p, 'launch_count', LimitedSpaceContainer(0, self.max_active_player_job))

        # Save active players' ``player_id``` & ``player_ckpt```.
        self.active_players_ids = [p.player_id for p in self.active_players]
        self.active_players_ckpts = [p.checkpoint_path for p in self.active_players]
        # Validate active players are unique by ``player_id``.`
        assert len(self.active_players_ids) == len(set(self.active_players_ids))

    def finish_job(self, job_info: dict) -> None:
        """
        Overview:
            Finish current job. Update active players' ``launch_count`` to release job space;
            Update shared payoff to record the game result.
        Arguments:
            - job_info (:obj:`dict`): A dict containing job result information.
        """
        # Update launch_count
        with self._active_players_lock:
            launch_player_id = job_info['launch_player']
            idx = self.active_players_ids.index(launch_player_id)
            self.active_players[idx].launch_count.release_space()
        # Update payoff to save job info
        # TODO(nyz) more fine-grained job info
        self.payoff.update(job_info)

    def run(self) -> None:
        """
        Overview:
            Start two threads: ``_launch_job_thread`` and ``_snapshot_thread``.
        """
        self._launch_job_thread.start()
        self._snapshot_thread.start()

    def close(self) -> None:
        """
        Overview:
            Close the league by setting ``_end_flag`` to True.
        """
        self._end_flag = True

    def _launch_job(self) -> None:
        """
        Overview:
            Launch a job if any active player has residual job space.
            Will run as a thread.
        """
        while not self._end_flag:
            with self._active_players_lock:
                # Check whether there are empty job launchers in any player.
                launch_counts = [p.launch_count.get_residual_space() for p in self.active_players]
                # Launch jobs.
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
        """
        Overview:
            Get info of the job which is to be launched to an active player. Called by ``_launch_job``.
        Arguments:
            - player (:obj:`ActivePlayer`): The active player to be launched a job.
        Returns:
            - job_info (:obj:`dict`): Job info. Should include keys ['lauch_player'].
        """
        raise NotImplementedError

    def _snapshot(self) -> None:
        """
        Overview:
            Find out active players that are trained enough, snapshot and mutate them.
            Will run as a thread.
        """
        time.sleep(int(0.5 * self.cfg.time_interval))
        while not self._end_flag:
            with self._active_players_lock:
                # Check whether there is an active player which is trained enough.
                flags = [p.is_trained_enough() for p in self.active_players]
                if sum(flags) != 0:
                    for idx, f in enumerate(flags):
                        if f:
                            player = self.active_players[idx]
                            # Snapshot
                            hp = player.snapshot()
                            self.save_checkpoint_fn(player.checkpoint_path, hp.checkpoint_path)
                            self.historical_players.append(hp)
                            self.payoff.add_player(hp)
                            # Mutate
                            self._mutate_player(player)
            time.sleep(self.cfg.time_interval)

    @abstractmethod
    def _mutate_player(self, player: ActivePlayer) -> None:
        """
        Overview:
            Players have the probability to be reset to supervised learning model parameters if trained enough.
            Called by ``self._snapshot``.
        Arguments:
            - player (:obj:`ActivePlayer`): the active player that may mutate
        """
        raise NotImplementedError

    def update_active_player(self, player_info: dict) -> None:
        """
        Overview:
            Update an active player's info.
        Arguments:
            - player_info (:obj:`dict`): Info dict of the player which is to be updated.
        """
        try:
            idx = self.active_players_ids.index(player_info['player_id'])
            player = self.active_players[idx]
            self._update_player(player, player_info)
        except ValueError:
            pass

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


league_mapping = {}


def register_league(name: str, league: type) -> None:
    """
    Overview:
        Add a new League class with its name to dict league_mapping, any subclass derived from
        BaseLeague must use this function to register in nervex system before instantiate.
    Arguments:
        - name (:obj:`str`): name of the new League class
        - learner (:obj:`type`): the new League class, should be subclass of BaseLeague
    """
    assert isinstance(name, str)
    assert issubclass(league, BaseLeague)
    league_mapping[name] = league


def create_league(cfg: EasyDict, *args) -> BaseLeague:
    """
    Overview:
        Given the key (league_type), create a new league instance if in league_mapping's values,
        or raise an KeyError. In other words, a derived league must first register then call ``create_league``
        to get the instance object.
    Arguments:
        - cfg (:obj:`EasyDict`): league config, necessary keys: [league.import_module, league.learner_type]
    Returns:
        - league (:obj:`BaseLeague`): the created new league, should be an instance of one of \
            league_mapping's values
    """
    import_module(cfg.league.import_names)
    league_type = cfg.league.league_type
    if league_type not in league_mapping.keys():
        raise KeyError("not support league type: {}".format(league_type))
    else:
        return league_mapping[league_type](cfg, *args)
