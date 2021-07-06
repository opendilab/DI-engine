import uuid
import copy
from abc import abstractmethod
from easydict import EasyDict
import os.path as osp

from ding.league.player import ActivePlayer, HistoricalPlayer, create_player
from ding.league.shared_payoff import create_payoff
from ding.utils import import_module, read_file, save_file, LockContext, LockContextType, LEAGUE_REGISTRY


class BaseLeague:
    """
    Overview:
        League, proposed by Google Deepmind AlphaStar. Can manage multiple players in one league.
    Interface:
        get_job_info, judge_snapshot, update_active_player, finish_job, save_checkpoint

    .. note::
        In ``__init__`` method, league would also initialized players as well(in ``_init_players`` method).
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: EasyDict) -> None:
        """
        Overview:
            Initialization method.
        Arguments:
            - cfg (:obj:`EasyDict`): League config.
        """
        self.cfg = cfg
        self.path_policy = cfg.path_policy
        self.league_uid = str(uuid.uuid1())
        self.active_players = []
        self.historical_players = []
        self.player_path = "./league"
        self.payoff = create_payoff(self.cfg.payoff)
        self._active_players_lock = LockContext(type_=LockContextType.THREAD_LOCK)
        self._init_players()

    def _init_players(self) -> None:
        """
        Overview:
            Initialize players (active & historical) in the league.
        """
        # Add different types of active players for each player category, according to ``cfg.active_players``.
        for cate in self.cfg.player_category:  # Player's category (Depends on the env)
            for k, n in self.cfg.active_players.items():  # Active player's type
                for i in range(n):  # This type's active player number
                    name = '{}_{}_{}_{}'.format(k, cate, i, self.league_uid)
                    ckpt_path = '{}_ckpt.pth'.format(name)
                    player = create_player(self.cfg, k, self.cfg[k], cate, self.payoff, ckpt_path, name, 0)
                    if self.cfg.use_pretrain:
                        self.save_checkpoint(
                            self.cfg.pretrain_checkpoint_path[cate], osp.join(self.path_policy, player.checkpoint_path)
                        )
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

        # Save active players' ``player_id``` & ``player_ckpt```.
        self.active_players_ids = [p.player_id for p in self.active_players]
        self.active_players_ckpts = [p.checkpoint_path for p in self.active_players]
        # Validate active players are unique by ``player_id``.
        assert len(self.active_players_ids) == len(set(self.active_players_ids))

    def get_job_info(self, player_id: str = None, eval_flag: bool = False) -> dict:
        """
        Overview:
            Get info dict of the job which is to be launched to an active player.
        Arguments:
            - player_id (:obj:`str`): The active player's id.
            - eval_flag (:obj:`bool`): Whether this is an evaluation job.
        Returns:
            - job_info (:obj:`dict`): Job info.
        ReturnsKeys:
            - necessary: ``launch_player`` (the active player)
        """
        if player_id is None:
            player_id = self.active_players_ids[0]
        with self._active_players_lock:
            idx = self.active_players_ids.index(player_id)
            player = self.active_players[idx]
            job_info = self._get_job_info(player, eval_flag)
            assert 'launch_player' in job_info.keys() and job_info['launch_player'] == player.player_id
        return job_info

    @abstractmethod
    def _get_job_info(self, player: ActivePlayer, eval_flag: bool = False) -> dict:
        """
        Overview:
            Real `get_job` method. Called by ``_launch_job``.
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
        with self._active_players_lock:
            idx = self.active_players_ids.index(player_id)
            player = self.active_players[idx]
            if player.is_trained_enough():
                # Snapshot
                hp = player.snapshot()
                self.save_checkpoint(
                    osp.join(self.path_policy, player.checkpoint_path), osp.join(self.path_policy, hp.checkpoint_path)
                )
                self.historical_players.append(hp)
                self.payoff.add_player(hp)
                # Mutate
                self._mutate_player(player)
                return True
            else:
                return False

    @abstractmethod
    def _mutate_player(self, player: ActivePlayer) -> None:
        """
        Overview:
            Players have the probability to mutate, e.g. Reset network parameters.
            Called by ``self.judge_snapshot``.
        Arguments:
            - player (:obj:`ActivePlayer`): The active player that may mutate.
        """
        raise NotImplementedError

    def update_active_player(self, player_info: dict) -> None:
        """
        Overview:
            Update an active player's info.
        Arguments:
            - player_info (:obj:`dict`): Info dict of the player which is to be updated.
        ArgumentsKeys:
            - necessary: `player_id`, `train_iteration`
        """
        try:
            idx = self.active_players_ids.index(player_info['player_id'])
            player = self.active_players[idx]
            return self._update_player(player, player_info)
        except ValueError as e:
            print(e)

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
        # TODO(nyz) more fine-grained job info
        self.payoff.update(job_info)

    @staticmethod
    def save_checkpoint(src_checkpoint, dst_checkpoint) -> None:
        '''
        Overview:
            Copy a checkpoint from path ``src_checkpoint`` to path ``dst_checkpoint``.
        Arguments:
            - src_checkpoint (:obj:`str`): Source checkpoint's path, e.g. s3://alphastar_fake_data/ckpt.pth
            - dst_checkpoint (:obj:`str`): Destination checkpoint's path, e.g. s3://alphastar_fake_data/ckpt.pth
        '''
        checkpoint = read_file(src_checkpoint)
        save_file(dst_checkpoint, checkpoint)


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
    import_module(cfg.get('import_names', []))
    return LEAGUE_REGISTRY.build(cfg.league_type, cfg=cfg, *args)
