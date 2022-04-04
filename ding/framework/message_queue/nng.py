import pynng
import logging
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
        self._finished = False

    def listen(self) -> None:
        self._sock = sock = Bus0()
        sock.listen(self.listen_to)
        sleep(0.1)  # Wait for peers to bind
        for contact in self.attach_to:
            sock.dial(contact)

    def publish(self, topic: str, data: bytes) -> None:
        if not self._finished:
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
                if self._finished:
                    break
                msg = self._sock.recv()
                # Use topic at the beginning of the message, so we don't need to call pickle.loads
                # when the current process is not subscribed to the topic.
                topic, payload = msg.split(b"::", maxsplit=1)
                return topic.decode(), payload
            except pynng.Timeout:
                logging.warning("Timeout on node {} when waiting for message from bus".format(self.listen_to))
            except pynng.Closed:
                if not self._finished:
                    logging.error("The socket was not closed under normal circumstances!")
            except Exception as e:
                logging.error("Meet exception when listening for new messages", e)

    def stop(self) -> None:
        finished = self._finished
        self._finished = True
        if not finished:
            self._sock.close()
            self._sock = None

    def __del__(self) -> None:
        self.stop()
