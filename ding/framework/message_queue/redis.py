import logging
from time import sleep
from typing import Tuple

import redis
from ding.framework.message_queue.mq import MQ
from ding.utils import MQ_REGISTRY


@MQ_REGISTRY.register("redis")
class RedisMQ(MQ):

    def __init__(self, redis_host: str, redis_port: int, **kwargs) -> None:
        """
        Overview:
            Connect distributed processes with redis
        Arguments:
            - redis_host (:obj:`str`): Redis server host.
            - redis_port (:obj:`int`): Redis server port.
        """
        self.host = redis_host
        self.port = redis_port if isinstance(redis_port, int) else int(redis_port)
        self.db = 0
        self._finished = False

    def listen(self) -> None:
        self._client = client = redis.Redis(host=self.host, port=self.port, db=self.db)
        self._sub = client.pubsub()

    def publish(self, topic: str, data: bytes) -> None:
        self._client.publish(topic, data)

    def subscribe(self, topic: str) -> None:
        self._sub.subscribe(topic)

    def unsubscribe(self, topic: str) -> None:
        self._sub.unsubscribe(topic)

    def recv(self) -> Tuple[str, bytes]:
        while True:
            if self._finished:
                return
            try:
                msg = self._sub.get_message(ignore_subscribe_messages=True)
                if msg is None:
                    sleep(0.001)
                    continue
                topic = msg["channel"].decode()
                data = msg["data"]
                return topic, data
            except (OSError, AttributeError, Exception) as e:
                logging.error("Meet exception when listening for new messages", e)

    def stop(self) -> None:
        self._finished = True
        self._sub.close()
        self._client.close()

    def __del__(self) -> None:
        self.stop()
