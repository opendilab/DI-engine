import os
import threading
import time
from threading import Thread

import numpy as np
import pytest
import yaml
from easydict import EasyDict

from sc2learner.data.online import ReplayBuffer

BATCH_SIZE = 8
PRODUCER_NUM = 16
CONSUMER_NUM = 4


@pytest.fixture(scope="function")
def setup_replay_buffer():
    with open(os.path.join(os.path.dirname(__file__), '../online/replay_buffer_default_config.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    return ReplayBuffer(cfg.replay_buffer)


def generate_data():
    ret = {
        'obs': np.random.randn(4),
    }
    p_weight = np.random.uniform()
    if p_weight < 1. / 3:
        pass  # no key 'priority'
    elif p_weight < 2. / 3:
        ret['priority'] = None
    else:
        ret['priority'] = np.random.uniform()

    return ret


class TestReplayBuffer:
    produce_count = 0

    def produce(self, id, replay_buffer):
        time.sleep(1)
        begin_time = time.time()
        count = 0
        while time.time() - begin_time < 20:
            t = np.random.randint(1, 6)
            time.sleep(t)
            print('[PRODUCER] thread {} use {} second to produce a data'.format(id, t))
            replay_buffer.push_data(generate_data())
            count += 1
        print('[PRODUCER] thread {} finish job, total produce {} data'.format(id, count))
        self.produce_count += count

    def consume(self, id, replay_buffer):
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
            print('[CONSUMER] thread {} iteration {} training finish'.format(id, iteration))
            # update
            info = {'priority': [], 'replay_buffer_id': [], 'replay_buffer_idx': []}
            for idx, d in enumerate(data):
                info['priority'].append(np.random.uniform() * 1.5)
                info['replay_buffer_id'].append(d['replay_buffer_id'])
                info['replay_buffer_idx'].append(d['replay_buffer_idx'])
            replay_buffer.update(info)
            print('[CONSUMER] thread {} iteration {} update finish'.format(id, iteration))

    def test(self, setup_replay_buffer):
        setup_replay_buffer._cache.debug = True
        p_threadings = [Thread(target=self.produce, args=(i, setup_replay_buffer)) for i in range(PRODUCER_NUM)]
        c_threadings = [Thread(target=self.consume, args=(i, setup_replay_buffer)) for i in range(CONSUMER_NUM)]
        for t in p_threadings:
            t.start()
        setup_replay_buffer.run()
        for t in c_threadings:
            t.start()

        for t in p_threadings:
            t.join()
        for t in c_threadings:
            t.join()
        setup_replay_buffer.close()
        time.sleep(1 + 0.1)
        assert (len(threading.enumerate()) <= 1)


if __name__ == '__main__':
    pytest.main(["test_replay_buffer.py"])
