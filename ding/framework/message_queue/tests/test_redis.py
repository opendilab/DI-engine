from time import sleep
import uuid
import pytest

from multiprocessing import Pool
from unittest.mock import Mock, patch
from threading import Thread
from ding.utils import WatchDog

from ding.framework.message_queue.redis import RedisMQ


def redis_main(i):
    node_id0 = uuid.uuid4().hex.encode()

    class MockRedis(Mock):

        def publish(self, topic, data):
            assert topic == "t"
            assert b"::" in data

        def pubsub(self):
            return MockPubSub()

    class MockPubSub(Mock):

        def get_message(self, **kwargs):
            return {"channel": b"t", "data": node_id0 + b"::data"}

    with patch("redis.Redis", MockRedis):
        host = "127.0.0.1"
        port = 6379
        mq = RedisMQ(redis_host=host, redis_port=port)
        mq.listen()
        if i == 0:
            mq._id = node_id0

            def send_message():
                for _ in range(5):
                    mq.publish("t", b"data")
                    sleep(0.1)

            def recv_message():
                # Should not receive any message
                mq.subscribe("t")
                print("RECV", mq.recv())

            send_thread = Thread(target=send_message, daemon=True)
            recv_thread = Thread(target=recv_message, daemon=True)
            send_thread.start()
            recv_thread.start()

            send_thread.join()

            watchdog = WatchDog(1)
            with pytest.raises(TimeoutError):
                watchdog.start()
                recv_thread.join()
            watchdog.stop()
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
