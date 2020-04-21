import os.path as osp
from collections import OrderedDict
from threading import Thread
import time

from sc2learner.utils import merge_dicts, read_config, LockContext
from sc2learner.league.player import ActivePlayer, MainPlayer, MainExploiter, LeagueExploiter, HistoricalPlayer
from sc2learner.league.shared_payoff import SharedPayoff

default_config = read_config(osp.join(osp.dirname(__file__), "league_default_config.yaml"))


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


class LeagueManager:
    """
    Overview: league training manager
    Interface: __init__, run, close, finish_match, update_active_player
    Note:
        launch_match_fn:
            Arguments:
                - launch_info (:obj:`dict`)
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
        match_info (:obj:`dict`)
            - home_id (:obj:`str`)
            - away_id (:obj:`str`)
            - result (:obj:`str`): `wins`, `draws`, `losses`
        player_info (:obj:`dict`)
            - player_id (:obj:`str`)
            - train_step (:obj:`int`)
    """
    def __init__(self, cfg, save_checkpoint_fn, load_checkpoint_fn, launch_match_fn):
        self.cfg = merge_dicts(default_config, cfg).league
        self.active_players = []
        self.historical_players = []
        self.payoff = SharedPayoff(self.cfg.payoff_decay, self.cfg.min_win_rate_games)
        self.max_active_player_match = self.cfg.max_active_player_match

        self.save_checkpoint_fn = save_checkpoint_fn
        self.load_checkpoint_fn = load_checkpoint_fn
        self.launch_match_fn = launch_match_fn
        self._active_players_lock = LockContext(lock_type='thread')
        self._launch_match_thread = Thread(target=self._launch_match)
        self._snapshot_thread = Thread(target=self._snapshot)
        self._end_flag = False

        self._init_league()

    def _init_league(self):
        player_map = {'main_player': MainPlayer, 'main_exploiter': MainExploiter, 'league_exploiter': LeagueExploiter}
        for r in self.cfg.race:
            for k, n in self.cfg.active_players.items():
                for i in range(n):
                    name = '{}_{}_{}'.format(k, r, i)  # e.g. main_player_zerg_0
                    ckpt_path = '{}_ckpt.pth'.format(name)
                    player = player_map[k](r, self.payoff, ckpt_path, name, **self.cfg[k])
                    self.active_players.append(player)
                    self.payoff.add_player(player)
                    self.save_checkpoint_fn(self.cfg.sl_checkpoint_path[r], player.checkpoint_path)

        # add sl player as the initial HistoricalPlayer
        if self.cfg.use_sl_init_historical:
            for r in self.cfg.race:
                name = '{}_{}_0_sl'.format('main_player', r)
                parent_name = '{}_{}_0'.format('main_player', r)
                hp = HistoricalPlayer(r, self.payoff, self.cfg.sl_checkpoint_path[r], name, parent_id=parent_name)
                self.historical_players.append(hp)
                self.payoff.add_player(hp)

        # register launch_count attribute for each active player
        for p in self.active_players:
            setattr(p, 'launch_count', LimitedSpaceContainer(0, self.max_active_player_match))

        # save active_players player_id
        self.active_players_ids = [p.player_id for p in self.active_players]
        self.active_players_ckpts = [p.checkpoint_path for p in self.active_players]
        # validate unique player_id
        assert len(self.active_players_ids) == len(set(self.active_players_ids))

    def run(self):
        self._launch_match_thread.start()
        self._snapshot_thread.start()

    def close(self):
        self._end_flag = True

    def _launch_match(self):
        while not self._end_flag:
            # check whether there is empty task launcher
            launch_counts = [0 for _ in range(len(self.active_players))]
            with self._active_players_lock:
                launch_counts = [p.launch_count.get_residual_space() for p in self.active_players]

                # launch match
                if sum(launch_counts) != 0:
                    for idx, c in enumerate(launch_counts):
                        for _ in range(c):
                            home = self.active_players[idx]
                            away = self.active_players[idx].get_match()
                            launch_info = {
                                'home_id': home.player_id,
                                'away_id': away.player_id,
                                'home_race': home.race,
                                'away_race': away.race,
                                'home_checkpoint_path': home.checkpoint_path,
                                'away_checkpoint_path': away.checkpoint_path,
                                'home_teacher_checkpoint_path': self.cfg.sl_checkpoint_path[home.race],
                                'away_teacher_checkpoint_path': self.cfg.sl_checkpoint_path[away.race],
                            }
                            self.launch_match_fn(launch_info)

            time.sleep(self.cfg.time_interval)

    def finish_match(self, match_info):
        # update launch_count
        with self._active_players_lock:
            home_id = match_info['home_id']
            idx = self.active_players_ids.index(home_id)
            self.active_players[idx].launch_count.release_space()
        # save match info
        # TODO(nyz) more fine-grained match info
        self.payoff.update(match_info)

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
                            info = {'sl_checkpoint_path': self.cfg.sl_checkpoint_path[player.race]}
                            result = player.mutate(info)
                            if result is not None:
                                self.load_checkpoint_fn(player.player_id, result)
                                self.save_checkpoint_fn(result, player.checkpoint_path)
            time.sleep(self.cfg.time_interval)

    def update_active_player(self, player_info):
        try:
            idx = self.active_players_ids.index(player_info['player_id'])
            self.active_players[idx].update_agent_step(player_info['train_step'])
        except ValueError:
            pass
