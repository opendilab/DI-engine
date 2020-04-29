import pytest
import time
import numpy as np
import threading
from threading import Thread
from multiprocessing import Queue
from multiprocessing.queues import Empty
from sc2learner.league.league_manager import LeagueManager

global BEGIN_COUNT, FINISH_COUNT
BEGIN_COUNT = 0
FINISH_COUNT = 0
SAVE_COUNT = 0


def save_checkpoint_fn(src_checkpoint, dst_checkpoint):
    global SAVE_COUNT
    t = np.random.uniform() + 0.5
    #time.sleep(t)
    print('save_checkpoint: src({})\tdst({})'.format(src_checkpoint, dst_checkpoint))
    SAVE_COUNT += 1


def load_checkpoint_fn(player_id, checkpoint_path):
    t = np.random.randint(2, 5)
    #time.sleep(t)
    print('load_checkpoint: player_id({})\tcheckpoint_path({})'.format(player_id, checkpoint_path))


class FakeMatchRunner:
    def __init__(self, random_match_result):
        self.queue = Queue(maxsize=10)
        self.random_match_result = random_match_result

    def launch_match(self, match_info):
        t = np.random.uniform() * 0.2 + 0.1
        time.sleep(t)
        thread = Thread(
            target=self.simulate_match, args=(match_info['home_id'], match_info['away_id'], self.random_match_result)
        )
        thread.start()

    def simulate_match(self, home_id, away_id, random_match_result):
        print('match begin: home({}) VS away({})'.format(home_id, away_id))
        global BEGIN_COUNT
        BEGIN_COUNT += 1
        t = np.random.randint(2, 4)
        time.sleep(t)
        self.queue.put({'home_id': home_id, 'away_id': away_id, 'result': random_match_result()})


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
            except Empty:
                continue
            finish_match(match_result)
            print(
                'match finish: home({}) {} away({})'.format(
                    match_result['home_id'], match_result['result'], match_result['away_id']
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


class TestLeagueManager:
    @pytest.mark.unittest
    def test_naive(self, random_match_result):
        match_runner = FakeMatchRunner(random_match_result)
        league_manager = LeagueManager({}, save_checkpoint_fn, load_checkpoint_fn, match_runner.launch_match)
        assert (len(league_manager.active_players) == 12)
        assert (len(league_manager.historical_players) == 3)
        active_player_ids = [p.player_id for p in league_manager.active_players]
        coordinator = FakeCoordinator(
            match_runner.queue, league_manager.finish_match, league_manager.update_active_player, active_player_ids
        )

        league_manager.run()
        coordinator.run()
        time.sleep(15)
        league_manager.close()
        time.sleep(league_manager.cfg.time_interval + 5)  # time_interval + simulate_match + receive_match time
        coordinator.close()
        time.sleep(5)
        assert BEGIN_COUNT == FINISH_COUNT
        assert (len(threading.enumerate()) <= 2), threading.enumerate()  # main thread + QueueFeederThread

    def test_snapshot_priority(self, random_match_result):
        global SAVE_COUNT
        SAVE_COUNT = 0
        match_runner = FakeMatchRunner(random_match_result)
        league_manager = LeagueManager({}, save_checkpoint_fn, load_checkpoint_fn, match_runner.launch_match)
        # fix mutate
        for p in league_manager.active_players:
            if hasattr(p, 'mutate_prob'):
                p.mutate_prob = 0.
        assert (len(league_manager.active_players) == 12)
        assert (len(league_manager.historical_players) == 3)
        active_player_ids = [p.player_id for p in league_manager.active_players]
        coordinator = FakeCoordinator(
            match_runner.queue, league_manager.finish_match, league_manager.update_active_player, active_player_ids
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
        assert SAVE_COUNT >= valid_count // 2 * 15 + 12  # count//2 * 15(12player+3mutate) + 12(init)
        assert (len(threading.enumerate()) <= 2), threading.enumerate()  # main thread + QueueFeederThread
