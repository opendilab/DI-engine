from time import sleep
import pytest

from multiprocessing import Pool
from unittest.mock import Mock, patch

from ding.framework.message_queue.redis import RedisMQ


def redis_main(i):

    class MockRedis(Mock):

        def publish(self, topic, data):
            assert topic == "t"
            assert data == b"data"

        def pubsub(self):
            return MockPubSub()

    class MockPubSub(Mock):

        def get_message(self, **kwargs):
            return {"channel": b"t", "data": b"data"}

    with patch("redis.Redis", MockRedis):
        host = "127.0.0.1"
        port = 6379
        mq = RedisMQ(redis_host=host, redis_port=port)
        mq.listen()
        if i == 0:
            for _ in range(5):
                mq.publish("t", b"data")
                sleep(0.3)
        else:
            mq.subscribe("t")
            topic, msg = mq.recv()
            assert topic == "t"
            assert msg == b"data"


@pytest.mark.unittest
@pytest.mark.execution_timeout(10)
def test_redis():
    with Pool(processes=2) as pool:
        pool.map(redis_main, range(2))
