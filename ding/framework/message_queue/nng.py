import pynng
from ditk import logging
from typing import List, Optional, Tuple
from pynng import Bus0
from time import sleep

from ding.framework.message_queue.mq import MQ
from ding.utils import MQ_REGISTRY


@MQ_REGISTRY.register("nng")
class NNGMQ(MQ):

    def __init__(self, listen_to: str, attach_to: Optional[List[str]] = None, **kwargs) -> None:
        """
        Overview:
            Connect distributed processes with nng
        Arguments:
            - listen_to (:obj:`Optional[List[str]]`): The node address to attach to.
            - attach_to (:obj:`Optional[List[str]]`): The node's addresses you want to attach to.
        """
        self.listen_to = listen_to
        self.attach_to = attach_to or []
        self._sock: Bus0 = None
        self._running = False

    def listen(self) -> None:
        self._sock = sock = Bus0()
        sock.listen(self.listen_to)
        sleep(0.1)  # Wait for peers to bind
        for contact in self.attach_to:
            sock.dial(contact)
        self._running = True

    def publish(self, topic: str, data: bytes) -> None:
        if self._running:
            topic += "::"
            data = topic.encode() + data
            self._sock.send(data)

    def subscribe(self, topic: str) -> None:
        return

    def unsubscribe(self, topic: str) -> None:
        return

    def recv(self) -> Tuple[str, bytes]:
        while True:
            try:
                if not self._running:
                    break
                msg = self._sock.recv()
                # Use topic at the beginning of the message, so we don't need to call pickle.loads
                # when the current process is not subscribed to the topic.
                topic, payload = msg.split(b"::", maxsplit=1)
                return topic.decode(), payload
            except pynng.Timeout:
                logging.warning("Timeout on node {} when waiting for message from bus".format(self.listen_to))
            except pynng.Closed:
                if self._running:
                    logging.error("The socket was not closed under normal circumstances!")
            except Exception as e:
                logging.error("Meet exception when listening for new messages", e)

    def stop(self) -> None:
        if self._running:
            self._running = False
            self._sock.close()
            self._sock = None

    def __del__(self) -> None:
        self.stop()
