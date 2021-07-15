import threading
import time
from threading import Thread

import numpy as np
import pytest

from ding.utils.data.structure import Cache


@pytest.mark.unittest
class TestCache:
    cache = Cache(16, 4, monitor_interval=1.0, _debug=True)
    send_count = 0
    produce_count = 0

    def producer(self, id):
        time.sleep(1)
        begin_time = time.time()
        count = 0
        while time.time() - begin_time < 20:
            t = np.random.randint(1, 6)
            time.sleep(t)
            print('[PRODUCER] thread {} use {} second to produce a data'.format(id, t))
            self.cache.push_data({'data': []})
            count += 1
        print('[PRODUCER] thread {} finish job, total produce {} data'.format(id, count))
        self.produce_count += count

    def consumer(self):
        for data in self.cache.get_cached_data_iter():
            self.send_count += 1
            print('[CONSUMER] cache send {}'.format(self.send_count))

    def test(self):
        producer_num = 8

        self.cache.run()
        threadings = [Thread(target=self.producer, args=(i, )) for i in range(producer_num)]
        for t in threadings:
            t.start()

        consumer_thread = Thread(target=self.consumer)
        consumer_thread.start()

        for t in threadings:
            t.join()

        # wait timeout mechanism to clear the cache
        time.sleep(4 + 1 + 0.1)

        assert (self.cache.remain_data_count == 0)
        assert (self.send_count == self.produce_count)

        self.cache.close()
        # wait the cache internal thread close and the consumer_thread get 'STOP' signal
        time.sleep(1 + 0.5)
        assert (not consumer_thread.is_alive())
