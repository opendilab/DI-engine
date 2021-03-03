import os
import time
from multiprocessing import Queue
from queue import Empty
from threading import Thread

import numpy as np
import pytest
import yaml
from easydict import EasyDict

from nervex.league import BaseLeague, register_league, create_league, ActivePlayer
from nervex.utils import deep_merge_dicts

global BEGIN_COUNT_BATTLE, BEGIN_COUNT_SOLO, FINISH_COUNT_BATTLE, FINISH_COUNT_SOLO
BEGIN_COUNT_BATTLE = 0
BEGIN_COUNT_SOLO = 0
FINISH_COUNT_BATTLE = 0
FINISH_COUNT_SOLO = 0
SAVE_COUNT = 0


class FakeLeague(BaseLeague):

    def __init__(self, cfg, *args, **kwargs):
        print("In fake league, we found arguments: {}, keyword arguments: {}".format(args, kwargs))
        super(FakeLeague, self).__init__(cfg)

    def _init_cfg(self, cfg: EasyDict) -> None:
        default_config = dict(
            league_type='one_vs_one',
            import_names=['nervex.league'],
            player_category=['cateA'],
            active_players='placeholder',
            use_pretrain=False,
            use_pretrain_init_historical=False,
            pretrain_checkpoint_path=dict(cateA='pretrain_checkpoint_solo.pth', ),
            payoff=dict(
                type='battle',
                decay=0.95,
                min_win_rate_games=4,
            ),
        )
        default_config = EasyDict(default_config)
        cfg = deep_merge_dicts(default_config, cfg)
        self.cfg = cfg.league
        # self.model_config = cfg.get('model', EasyDict())

    def _get_job_info(self, player):
        return {
            'launch_player': player.player_id,
            'player_id': [player.player_id, player.player_id],
        }

    def _mutate_player(self, player):
        # info = {'pretrain_checkpoint_path': 'pretrain_path_placeholder'}
        # result = player.mutate(info)
        # if result is not None:
        #     self.load_checkpoint_fn(player.player_id, result)
        #     self.save_checkpoint_fn(result, player.checkpoint_path)
        pass

    def _update_player(self, player, player_info):
        assert isinstance(player, ActivePlayer)
        player.total_agent_step = player_info['train_step']


@pytest.fixture(scope='function')
def setup_config():
    with open(os.path.join(os.path.dirname(__file__), 'league_test_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    return cfg


def save_checkpoint_fn(src_checkpoint, dst_checkpoint):
    global SAVE_COUNT
    t = np.random.uniform() + 0.5
    # time.sleep(t)
    print('save_checkpoint: src({})\tdst({})'.format(src_checkpoint, dst_checkpoint))
    SAVE_COUNT += 1


def load_checkpoint_fn(player_id, checkpoint_path):
    t = np.random.randint(2, 5)
    # time.sleep(t)
    print('load_checkpoint: player_id({})\tcheckpoint_path({})'.format(player_id, checkpoint_path))


class FakeBattleMatchRunner:

    def __init__(self, random_job_result):
        self.queue = Queue(maxsize=10)
        self.random_job_result = random_job_result

    def launch_match(self, match_info):
        print('match_info', match_info)
        t = np.random.uniform() * 0.2 + 0.1
        time.sleep(t)
        thread = Thread(target=self._simulate_match, args=(match_info, self.random_job_result))
        thread.start()

    def _simulate_match(self, match_info, random_job_result):
        home_id, away_id = match_info['player_id']
        print('match begin: home({}) VS away({})'.format(home_id, away_id))
        global BEGIN_COUNT_BATTLE
        BEGIN_COUNT_BATTLE += 1
        t = np.random.randint(2, 4)
        time.sleep(t)
        self.queue.put(
            {
                'player_id': [home_id, away_id],
                'launch_player': match_info['launch_player'],
                'result': [[random_job_result()]],
            }
        )


class FakeSoloMatchRunner:

    def __init__(self, random_job_result):
        self.queue = Queue(maxsize=10)
        self.random_job_result = random_job_result

    def launch_match(self, match_info):
        print('match_info', match_info)
        t = np.random.uniform() * 0.2 + 0.1
        time.sleep(t)
        thread = Thread(target=self._simulate_match, args=(match_info, self.random_job_result))
        thread.start()

    def _simulate_match(self, match_info, random_job_result):
        player_id = match_info['player_id'][0]
        print('match begin: player({})'.format(player_id))
        global BEGIN_COUNT_BATTLE
        BEGIN_COUNT_BATTLE += 1
        t = np.random.randint(2, 4)
        time.sleep(t)
        self.queue.put(
            {
                'player_id': [player_id],
                'launch_player': match_info['launch_player'],
                'result': [[random_job_result()]],
            }
        )


class FakeCoordinator:

    def __init__(self, league_type, queue, finish_match, update_agent_step, player_ids):
        if league_type == 'battle':
            self.receive_match_thread = Thread(target=self.receive_match_battle, args=(queue, finish_match))
        elif league_type == 'solo':
            self.receive_match_thread = Thread(target=self.receive_match_solo, args=(queue, finish_match))
        self.update_train_step_thread = Thread(target=self.update_train_step, args=(update_agent_step,))
        self.player_ids = player_ids
        self.one_phase_steps = int(2e3)
        self._end_flag = False

    def run(self):
        self.receive_match_thread.start()
        self.update_train_step_thread.start()

    def close(self):
        self._end_flag = True

    def receive_match_battle(self, queue, finish_match):
        global FINISH_COUNT_BATTLE
        while not self._end_flag:
            try:
                match_result = queue.get(timeout=1)
            except Empty:
                continue
            finish_match(match_result)
            print(
                'match finish: home({}) {} away({})'.format(
                    match_result['player_id'][0], match_result['result'][0][0], match_result['player_id'][1]
                )
            )
            FINISH_COUNT_BATTLE += 1
            time.sleep(0.1)

    def receive_match_solo(self, queue, finish_match):
        global FINISH_COUNT_BATTLE
        while not self._end_flag:
            try:
                match_result = queue.get(timeout=1)
            except Empty:
                continue
            finish_match(match_result)
            print('match finish: player({}) {}'.format(match_result['player_id'][0], match_result['result'][0][0]))
            FINISH_COUNT_BATTLE += 1
            time.sleep(0.1)

    def update_train_step(self, update_agent_step):
        self.update_count = 0
        while not self._end_flag:
            time.sleep(2)
            self.update_count += 1
            for player_id in self.player_ids:
                update_agent_step({'player_id': player_id, 'train_step': self.update_count * self.one_phase_steps})


class TestFakeLeague:

    # @pytest.mark.unittest
    def test_naive(self, random_job_result, setup_config):
        match_runner = FakeBattleMatchRunner(random_job_result)
        register_league('fake', FakeLeague)
        league = create_league(setup_config, save_checkpoint_fn, load_checkpoint_fn, match_runner.launch_match)
        assert (len(league.active_players) == 12)
        assert (len(league.historical_players) == 3)
        active_player_ids = [p.player_id for p in league.active_players]
        coordinator = FakeCoordinator(
            'battle', match_runner.queue, league.finish_job, league.update_active_player, active_player_ids
        )

        league.run()
        coordinator.run()
        time.sleep(15)
        league.close()
        time.sleep(league.cfg.time_interval + 5)  # time_interval + simulate_match + receive_match time
        coordinator.close()
        time.sleep(5)
        assert BEGIN_COUNT_BATTLE == FINISH_COUNT_BATTLE

    # TODO(zlx): priority lock
    def test_snapshot_priority(self, random_job_result, setup_config):
        global SAVE_COUNT
        SAVE_COUNT = 0
        match_runner = FakeBattleMatchRunner(random_job_result)
        league = create_league(setup_config, save_checkpoint_fn, load_checkpoint_fn, match_runner.launch_match)
        # fix mutate
        for p in league.active_players:
            if hasattr(p, 'mutate_prob'):
                p.mutate_prob = 0.
        assert (len(league.active_players) == 12)
        assert (len(league.historical_players) == 3)
        active_player_ids = [p.player_id for p in league.active_players]
        coordinator = FakeCoordinator(
            'battle', match_runner.queue, league.finish_job, league.update_active_player, active_player_ids
        )

        league.run()
        coordinator.run()
        time.sleep(12.5)
        league.close()
        valid_count = coordinator.update_count
        time.sleep(league.cfg.time_interval + 5)  # time_interval + simulate_match + receive_match time
        coordinator.close()
        time.sleep(2)
        assert BEGIN_COUNT_BATTLE == FINISH_COUNT_BATTLE
        # assert SAVE_COUNT >= valid_count // 2 * 15 + 12  # count//2 * 15(12player+3mutate) + 12(init)
