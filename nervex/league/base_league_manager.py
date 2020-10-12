import os.path as osp
import time
from abc import ABC, abstractmethod
from threading import Thread

from nervex.league.player import MainPlayer, MainExploiter, LeagueExploiter, HistoricalPlayer
from nervex.league.shared_payoff import SharedPayoff
from nervex.utils import merge_dicts, read_config, LockContext, import_module

default_config = read_config(osp.join(osp.dirname(__file__), "league_manager_default_config.yaml"))


class LimitedSpaceContainer:

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        assert (max_val > min_val)
        self.cur = self.min_val

    def get_residual_space(self):
        ret = self.max_val - self.cur
        self.cur = self.max_val
        return ret

    def release_space(self):
        self.cur = max(self.min_val, self.cur - 1)


class BaseLeagueManager(ABC):
    """
    Overview: league training manager
    Interface: __init__, run, close, finish_task, update_active_player
    Note:
        launch_task_fn:
            Arguments:
                - task_info (:obj:`dict`)
                    - home_id (:obj:`str`)
                    - away_id (:obj:`str`)
                    - home_race (:obj:`str`)
                    - away_race (:obj:`str`)
                    - home_checkpoint_path (:obj:`str`)
                    - away_checkpoint_path (:obj:`str`)
                    - home_teacher_checkpoint_path (:obj:`str`)
                    - away_teacher_checkpoint_path (:obj:`str`)
        save_checkpoint_fn:
            Arguments:
                - src_checkpoint (:obj:`str`): src must be a existing path
                - dst_checkpoint (:obj:`str`)
        load_checkpoint_fn:
            Arguments:
                - player_id: (:obj:`str`)
                - checkpoint_path: (:obj:`str`)
        task_info (:obj:`dict`)
            - home_id (:obj:`str`)
            - away_id (:obj:`str`)
            - result (:obj:`str`): `wins`, `draws`, `losses`
        player_info (:obj:`dict`)
            - player_id (:obj:`str`)
            - train_step (:obj:`int`)
    """

    def __init__(self, cfg, save_checkpoint_fn, load_checkpoint_fn, launch_task_fn):
        cfg = merge_dicts(default_config, cfg)
        self.cfg = cfg.league
        self.model_config = cfg.model
        self.active_players = []
        self.historical_players = []
        self.payoff = SharedPayoff(self.cfg.payoff_decay, self.cfg.min_win_rate_games)
        self.max_active_player_task = self.cfg.max_active_player_task

        self.save_checkpoint_fn = save_checkpoint_fn
        self.load_checkpoint_fn = load_checkpoint_fn
        self.launch_task_fn = launch_task_fn
        self._active_players_lock = LockContext(lock_type='thread')
        self._launch_task_thread = Thread(target=self._launch_task)
        self._snapshot_thread = Thread(target=self._snapshot)
        self._end_flag = False

        self._init_league()

    def _init_league(self):
        player_map = {'main_player': MainPlayer, 'main_exploiter': MainExploiter, 'league_exploiter': LeagueExploiter}
        for r in self.cfg.player_category:
            for k, n in self.cfg.active_players.items():
                for i in range(n):
                    name = '{}_{}_{}'.format(k, r, i)
                    ckpt_path = '{}_ckpt.pth'.format(name)
                    player = player_map[k](r, self.payoff, ckpt_path, name, **self.cfg[k])
                    self.active_players.append(player)
                    self.payoff.add_player(player)

        # add pretrain player as the initial HistoricalPlayer
        if self.cfg.use_pretrain_init_historical:
            for r in self.cfg.player_category:
                name = '{}_{}_0_pretrain'.format('main_player', r)
                parent_name = '{}_{}_0'.format('main_player', r)
                hp = HistoricalPlayer(r, self.payoff, self.cfg.pretrain_checkpoint_path[r], name, parent_id=parent_name)
                self.historical_players.append(hp)
                self.payoff.add_player(hp)

        # register launch_count attribute for each active player
        for p in self.active_players:
            setattr(p, 'launch_count', LimitedSpaceContainer(0, self.max_active_player_task))

        # save active_players player_id
        self.active_players_ids = [p.player_id for p in self.active_players]
        self.active_players_ckpts = [p.checkpoint_path for p in self.active_players]
        # validate unique player_id
        assert len(self.active_players_ids) == len(set(self.active_players_ids))

    def run(self):
        self._launch_task_thread.start()
        self._snapshot_thread.start()

    def close(self):
        self._end_flag = True

    def _launch_task(self):
        while not self._end_flag:
            # check whether there is empty task launcher
            with self._active_players_lock:
                launch_counts = [p.launch_count.get_residual_space() for p in self.active_players]

                # launch task
                if sum(launch_counts) != 0:
                    for idx, c in enumerate(launch_counts):
                        for _ in range(c):
                            player = self.active_players[idx]
                            task_info = self._get_task_info(player)
                            assert 'launch_player' in task_info.keys(
                            ) and task_info['launch_player'] == player.player_id
                            self.launch_task_fn(task_info)

            time.sleep(self.cfg.time_interval)

    def finish_task(self, task_info):
        # update launch_count
        with self._active_players_lock:
            launch_player = task_info['launch_player']
            idx = self.active_players_ids.index(launch_player)
            self.active_players[idx].launch_count.release_space()
        # save task info
        # TODO(nyz) more fine-grained task info
        self.payoff.update(task_info)

    def _snapshot(self):
        time.sleep(int(0.5 * self.cfg.time_interval))
        while not self._end_flag:
            # check whether there is a active player which is trained enough
            with self._active_players_lock:
                flags = [p.is_trained_enough() for p in self.active_players]

                # snapshot and mutate
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

    def update_active_player(self, player_info):
        try:
            idx = self.active_players_ids.index(player_info['player_id'])
            player = self.active_players[idx]
            self._update_player(player, player_info)
        except ValueError:
            pass

    @abstractmethod
    def _get_task_info(self, player):
        raise NotImplementedError

    @abstractmethod
    def _mutate_player(self, player):
        raise NotImplementedError

    @abstractmethod
    def _update_player(self, player, player_info):
        raise NotImplementedError


league_mapping = {}


def register_league(name: str, league: type) -> None:
    assert isinstance(name, str)
    assert issubclass(league, BaseLeagueManager)
    league_mapping[name] = league


def create_league(cfg: dict, *args) -> BaseLeagueManager:
    import_module(cfg.league.import_names)
    league_type = cfg.league.league_type
    if league_type not in league_mapping.keys():
        raise KeyError("not support league type: {}".format(league_type))
    else:
        return league_mapping[league_type](cfg, *args)
