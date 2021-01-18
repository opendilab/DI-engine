import os
import random
import threading
import time
from threading import Thread
from typing import List
import numpy as np
import pytest
import pickle
# import cProfile
# import pstats
# import io
# from pstats import SortKey

from nervex.data import ReplayBuffer
from nervex.utils import read_config

BATCH_SIZE = 8
PRODUCER_NUM = 16
CONSUMER_NUM = 4
LASTING_TIME = 5
np.random.seed(1)


@pytest.fixture(scope="function")
def setup_config():
    path = os.path.join(os.path.dirname(__file__), '../replay_buffer_default_config.yaml')
    cfg = read_config(path)
    cfg.replay_buffer.agent.enable_track_used_data = True
    return cfg


@pytest.fixture(scope="function")
def setup_demo_config():
    path = os.path.join(os.path.dirname(__file__), '../replay_buffer_with_demonstration_config.yaml')
    cfg = read_config(path)
    cfg.replay_buffer.agent.enable_track_used_data = True
    cfg.replay_buffer.demo.enable_track_used_data = True
    cfg.replay_buffer.sample_ratio.agent = 0.5
    cfg.replay_buffer.sample_ratio.demo = 0.5
    return cfg


def generate_data() -> dict:
    ret = {'obs': np.random.randn(4), 'data_push_length': 8}
    p_weight = np.random.uniform()
    if p_weight < 1. / 3:
        pass  # no key 'priority'
    elif p_weight < 2. / 3:
        ret['priority'] = None
    else:
        ret['priority'] = np.random.uniform()
    return ret


def generate_data_list(count: int) -> List[dict]:
    return [generate_data() for _ in range(0, count)]


class TestReplayBuffer:
    produce_count = 0
    global_data = []

    def produce(
            self,
            id_,
            replay_buffer,
            buffer_name: list = ['agent', 'agent'],
            pressure: int = 1,
            lasting_time: int = LASTING_TIME
    ) -> None:
        time.sleep(1)
        begin_time = time.time()
        count = 0
        while time.time() - begin_time < lasting_time:
            duration = np.random.randint(1, 4) / pressure
            time.sleep(duration)
            if np.random.randint(0, 100) > 50:
                print('[PRODUCER] thread {} use {} second to produce 1 data'.format(id_, duration))
                replay_buffer.push_data(generate_data(), buffer_name[0])
                count += 1
            else:
                data_count = np.random.randint(2, 5)
                print(
                    '[PRODUCER] thread {} use {} second to produce a list of {} data'.format(id_, duration, data_count)
                )
                replay_buffer.push_data(generate_data_list(data_count), buffer_name[1])
                count += data_count
        print('[PRODUCER] thread {} finish job, total produce {} data'.format(id_, count))
        self.produce_count += count

    def consume(self, id_, replay_buffer, pressure: int = 1, lasting_time: int = LASTING_TIME + 5) -> None:
        time.sleep(1)
        begin_time = time.time()
        iteration = 0
        while time.time() - begin_time < lasting_time:
            while True:
                data = replay_buffer.sample(BATCH_SIZE, 0)
                if data is not None:
                    assert len(data) == BATCH_SIZE
                    self.global_data += data
                    break
                else:
                    time.sleep(2 / pressure)
            time.sleep(0.5 / pressure)
            iteration += 1
            print('[CONSUMER] thread {} iteration {} training finish'.format(id_, iteration))
            # update
            info = {'priority': [], 'replay_unique_id': [], 'replay_buffer_idx': []}
            for idx, d in enumerate(data):
                info['priority'].append(np.random.uniform() * 1.5)
                info['replay_unique_id'].append(d['replay_unique_id'])
                info['replay_buffer_idx'].append(d['replay_buffer_idx'])
            replay_buffer.update(info)
        print('[CONSUMER] thread {} iteration {} update finish'.format(id_, iteration))

    @pytest.mark.unittest
    def test_single_buffer(self, setup_config):
        # pr = cProfile.Profile()
        # pr.enable()

        self.global_data = []
        os.popen('rm -rf log*')
        setup_replay_buffer = ReplayBuffer(setup_config.replay_buffer)
        setup_replay_buffer._cache.debug = True
        produce_threads = [Thread(target=self.produce, args=(i, setup_replay_buffer)) for i in range(PRODUCER_NUM)]
        consume_threads = [Thread(target=self.consume, args=(i, setup_replay_buffer)) for i in range(CONSUMER_NUM)]
        for t in produce_threads:
            t.start()
        setup_replay_buffer.run()
        for t in consume_threads:
            t.start()

        for t in produce_threads:
            t.join()
        for t in consume_threads:
            t.join()
        used_data = setup_replay_buffer.used_data()
        count = setup_replay_buffer.count()
        setup_replay_buffer.push_data({'data': np.random.randn(4)})
        setup_replay_buffer.close()
        time.sleep(1 + 0.5)
        assert (len(threading.enumerate()) <= 3)
        os.popen('rm -rf log*')

        # pr.disable()
        # s = io.StringIO()
        # ps = pstats.Stats(pr, stream=s)
        # ps.print_stats()
        # with open("./replay_buffer_profile.txt", "w") as f:
        #     f.write(s.getvalue())

    @pytest.mark.unittest
    def test_double_buffer(self, setup_demo_config):
        # pr = cProfile.Profile()
        # pr.enable()

        os.popen('rm -rf log*')

        self.global_data = []
        demo_data_list = generate_data_list(50)
        with open("demonstration_data.pkl", "wb") as f:
            pickle.dump(demo_data_list, f)
        setup_replay_buffer = ReplayBuffer(setup_demo_config.replay_buffer)
        setup_replay_buffer._cache.debug = True
        os.popen("rm -rf demonstration_data.pkl")

        produce_threads = [
            Thread(target=self.produce, args=(i, setup_replay_buffer, ['agent', 'demo'])) for i in range(PRODUCER_NUM)
        ]
        consume_threads = [Thread(target=self.consume, args=(i, setup_replay_buffer)) for i in range(CONSUMER_NUM)]
        for t in produce_threads:
            t.start()
        setup_replay_buffer.run()
        for t in consume_threads:
            t.start()

        for t in produce_threads:
            t.join()
        for t in consume_threads:
            t.join()
        agent_used_data = setup_replay_buffer.used_data('agent')
        demo_used_data = setup_replay_buffer.used_data('demo')
        agent_count = setup_replay_buffer.count('agent')
        demo_count = setup_replay_buffer.count('demo')
        setup_replay_buffer.push_data({'data': np.random.randn(4)}, 'agent')
        setup_replay_buffer.close()
        time.sleep(1 + 0.5)
        assert (len(threading.enumerate()) <= 4), threading.enumerate()

        agent_count, demo_count = 0, 0
        for data in self.global_data:
            if 'agent' in data['replay_unique_id']:
                agent_count += 1
            elif 'demo' in data['replay_unique_id']:
                demo_count += 1
        assert 0.8 < agent_count / demo_count < 1.25

        os.popen('rm -rf log*')

        # pr.disable()
        # s = io.StringIO()
        # ps = pstats.Stats(pr, stream=s)
        # ps.print_stats()
        # with open("./replay_buffer_profile.txt", "w") as f:
        #     f.write(s.getvalue())

    @pytest.mark.unittest
    def test_serial(self, setup_config):
        # pr = cProfile.Profile()
        # pr.enable()

        os.popen('rm -rf log*')
        replay_buffer = ReplayBuffer(setup_config.replay_buffer)
        replay_buffer._cache.debug = True

        begin_time = time.time()
        total_produce_count, total_consume_count = 0, 0
        iteration = 0
        while time.time() - begin_time < 30:
            # produce
            produce_count = 0
            produce_begin_time = time.time()
            while produce_count < 20000:
                if np.random.randint(0, 100) > 80:
                    replay_buffer.push_data(generate_data())
                    produce_count += 1
                else:
                    data_count = np.random.randint(10, 50)
                    replay_buffer.push_data(generate_data_list(data_count))
                    produce_count += data_count
            print(
                '[PRODUCER] produce {} data, using {} seconds'.format(produce_count,
                                                                      time.time() - produce_begin_time)
            )
            total_produce_count += produce_count
            # consume
            consume_count = 0
            consume_begin_time = time.time()
            while consume_count < 3000:
                data = replay_buffer.sample(BATCH_SIZE, 0)
                if data is None:
                    break
                assert (len(data) == BATCH_SIZE)
                iteration += 1
                consume_count += BATCH_SIZE
                # update replay buffer
                info = {'priority': [], 'replay_unique_id': [], 'replay_buffer_idx': []}
                for idx, d in enumerate(data):
                    info['priority'].append(np.random.uniform() * 1.5)
                    info['replay_unique_id'].append(d['replay_unique_id'])
                    info['replay_buffer_idx'].append(d['replay_buffer_idx'])
                replay_buffer.update(info)
            print(
                '[CONSUMER] iteration {} training finish, sampling {} data, using {} seconds'.format(
                    iteration, consume_count,
                    time.time() - consume_begin_time
                )
            )
            total_consume_count += consume_count
        print('[PRODUCER] finish job, total produce {} data'.format(total_produce_count))
        print('[CONSUMER] finish job, total consume {} data'.format(total_consume_count))

        os.popen('rm -rf log*')

        # pr.disable()
        # s = io.StringIO()
        # ps = pstats.Stats(pr, stream=s)
        # ps.print_stats()
        # with open("./replay_buffer_profile_serial.txt", "w") as f:
        #     f.write(s.getvalue())
