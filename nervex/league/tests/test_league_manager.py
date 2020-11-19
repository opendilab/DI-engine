import os
import threading
import time
from multiprocessing import Queue
# from multiprocessing.queues import Empty
from threading import Thread

import numpy as np
import pytest
import yaml
from easydict import EasyDict

from nervex.league import BaseLeagueManager

global BEGIN_COUNT, FINISH_COUNT
BEGIN_COUNT = 0
FINISH_COUNT = 0
SAVE_COUNT = 0


class FakeLeagueManager(BaseLeagueManager):

    def _get_job_info(self, player):
        return {
            'launch_player': player.player_id,
            'player_id': [player.player_id, player.player_id],
        }

    def _mutate_player(self, player):
        info = {'pretrain_checkpoint_path': 'pretrain_path_placeholder'}
        result = player.mutate(info)
        if result is not None:
            self.load_checkpoint_fn(player.player_id, result)
            self.save_checkpoint_fn(result, player.checkpoint_path)

    def _update_player(self, player, player_info):
        pass


@pytest.fixture(scope='function')
def setup_config():
    with open(os.path.join(os.path.dirname(__file__), 'league_manager_test_config.yaml')) as f:
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


class FakeMatchRunner:

    def __init__(self, random_job_result):
        self.queue = Queue(maxsize=10)
        self.random_job_result = random_job_result

    def launch_match(self, match_info):
        print('match_info', match_info)
        t = np.random.uniform() * 0.2 + 0.1
        time.sleep(t)
        thread = Thread(target=self.simulate_match, args=(match_info, self.random_job_result))
        thread.start()

    def simulate_match(self, match_info, random_job_result):
        home_id, away_id = match_info['player_id']
        print('match begin: home({}) VS away({})'.format(home_id, away_id))
        global BEGIN_COUNT
        BEGIN_COUNT += 1
        t = np.random.randint(2, 4)
        time.sleep(t)
        self.queue.put(
            {
                'player_id': [home_id, away_id],
                'launch_player': match_info['launch_player'],
                'result': [[random_job_result()]],
            }
        )


class FakeCoordinator:

    def __init__(self, queue, finish_match, update_agent_step, player_ids):
        self.receive_match_thread = Thread(target=self.receive_match, args=(queue, finish_match))
        self.update_train_step_thread = Thread(target=self.update_train_step, args=(update_agent_step, ))
        self.player_ids = player_ids
        self.one_phase_steps = int(2e3)
        self._end_flag = False

    def run(self):
        self.receive_match_thread.start()
        self.update_train_step_thread.start()

    def close(self):
        self._end_flag = True

    def receive_match(self, queue, finish_match):
        global FINISH_COUNT
        while not self._end_flag:
            try:
                match_result = queue.get(timeout=1)
            except Exception:  # except Empty:
                continue
            finish_match(match_result)
            print(
                'match finish: home({}) {} away({})'.format(
                    match_result['player_id'][0], match_result['result'][0][0], match_result['player_id'][1]
                )
            )
            FINISH_COUNT += 1
            time.sleep(0.1)

    def update_train_step(self, update_agent_step):
        self.update_count = 0
        while not self._end_flag:
            time.sleep(2)
            self.update_count += 1
            for player_id in self.player_ids:
                update_agent_step({'player_id': player_id, 'train_step': self.update_count * self.one_phase_steps})


class TestFakeLeagueManager:

    @pytest.mark.unittest
    def test_naive(self, random_job_result, setup_config):
        match_runner = FakeMatchRunner(random_job_result)
        league_manager = FakeLeagueManager(
            setup_config, save_checkpoint_fn, load_checkpoint_fn, match_runner.launch_match
        )
        assert (len(league_manager.active_players) == 12)
        assert (len(league_manager.historical_players) == 3)
        active_player_ids = [p.player_id for p in league_manager.active_players]
        coordinator = FakeCoordinator(
            match_runner.queue, league_manager.finish_job, league_manager.update_active_player, active_player_ids
        )

        league_manager.run()
        coordinator.run()
        time.sleep(15)
        league_manager.close()
        time.sleep(league_manager.cfg.time_interval + 5)  # time_interval + simulate_match + receive_match time
        coordinator.close()
        time.sleep(5)
        assert BEGIN_COUNT == FINISH_COUNT
        assert (len(threading.enumerate()) <= 3), threading.enumerate()  # main thread + QueueFeederThread

    def test_snapshot_priority(self, random_job_result, setup_config):
        global SAVE_COUNT
        SAVE_COUNT = 0
        match_runner = FakeMatchRunner(random_job_result)
        league_manager = FakeLeagueManager(
            setup_config, save_checkpoint_fn, load_checkpoint_fn, match_runner.launch_match
        )
        # fix mutate
        for p in league_manager.active_players:
            if hasattr(p, 'mutate_prob'):
                p.mutate_prob = 0.
        assert (len(league_manager.active_players) == 12)
        assert (len(league_manager.historical_players) == 3)
        active_player_ids = [p.player_id for p in league_manager.active_players]
        coordinator = FakeCoordinator(
            match_runner.queue, league_manager.finish_job, league_manager.update_active_player, active_player_ids
        )

        league_manager.run()
        coordinator.run()
        time.sleep(12.5)
        league_manager.close()
        valid_count = coordinator.update_count
        time.sleep(league_manager.cfg.time_interval + 5)  # time_interval + simulate_match + receive_match time
        coordinator.close()
        time.sleep(2)
        assert BEGIN_COUNT == FINISH_COUNT
        # TODO(zlx): why
        # assert SAVE_COUNT >= valid_count // 2 * 15 + 12  # count//2 * 15(12player+3mutate) + 12(init)
        assert (len(threading.enumerate()) <= 2), threading.enumerate()  # main thread + QueueFeederThread
