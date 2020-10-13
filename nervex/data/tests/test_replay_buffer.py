import os
import random
import threading
import time
from threading import Thread
from typing import List

import numpy as np
import pytest
import yaml
from easydict import EasyDict

from nervex.data.online import ReplayBuffer

BATCH_SIZE = 8
PRODUCER_NUM = 16
CONSUMER_NUM = 4


@pytest.fixture(scope="function")
def setup_replay_buffer():
    with open(os.path.join(os.path.dirname(__file__), '../online/replay_buffer_default_config.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    return ReplayBuffer(cfg.replay_buffer)


def generate_data() -> dict:
    ret = {'obs': np.random.randn(4), 'data_push_length': 1}
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


@pytest.mark.unittest
class TestReplayBuffer:
    produce_count = 0

    def produce(self, id_, replay_buffer):
        time.sleep(1)
        begin_time = time.time()
        count = 0
        while time.time() - begin_time < 20:
            duration = np.random.randint(1, 6)
            time.sleep(duration)

            if random.randint(0, 100) > 50:
                print('[PRODUCER] thread {} use {} second to produce a data'.format(id_, duration))
                replay_buffer.push_data(generate_data())
                count += 1
            else:
                data_count = random.randint(2, 5)
                print(
                    '[PRODUCER] thread {} use {} second to produce a list of {} data'.format(id_, duration, data_count)
                )
                replay_buffer.push_data(generate_data_list(data_count))
                count += data_count

        print('[PRODUCER] thread {} finish job, total produce {} data'.format(id_, count))
        self.produce_count += count

    def consume(self, id_, replay_buffer):
        time.sleep(1)
        begin_time = time.time()
        iteration = 0
        while time.time() - begin_time < 25:
            while True:
                data = replay_buffer.sample(BATCH_SIZE)
                if data is not None:
                    assert (len(data) == BATCH_SIZE)
                    break
                else:
                    time.sleep(2)
            time.sleep(0.5)
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

    def test(self, setup_replay_buffer):
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
        setup_replay_buffer.close()
        time.sleep(1 + 0.5)
        assert (len(threading.enumerate()) <= 1)


if __name__ == '__main__':
    pytest.main(["test_replay_buffer.py"])
