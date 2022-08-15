from typing import Tuple, Optional


class MQ:
    """
    Overview:
        Abstract basic mq class.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Overview:
            The __init__ method of the inheritance must support the extra kwargs parameter.
        """
        pass

    def listen(self) -> None:
        """
        Overview:
            Bind to local socket or connect to third party components.
        """
        raise NotImplementedError

    def publish(self, topic: str, data: bytes) -> None:
        """
        Overview:
            Send data to mq.
        Arguments:
            - topic (:obj:`str`): Topic.
            - data (:obj:`bytes`): Payload data.
        """
        raise NotImplementedError

    def subscribe(self, topic: str, fn: Optional[callable] = None, is_once: Optional[bool] = False) -> None:
        """
        Overview:
            Subscribe to the topic.
        Arguments:
            - topic (:obj:`str`): Topic
            - fn (:obj:`Optional[callable]`): The message handler, if the communication library
                implements event_loop, it can bypass Parallel() and calling this function by itself.
            - is_once (:obj:`bool`):  Whether Topic will only be called once.
        """
        raise NotImplementedError

    def unsubscribe(self, topic: str) -> None:
        """
        Overview:
            Unsubscribe from the topic.
        Arguments:
            - topic (:obj:`str`): Topic
        """
        raise NotImplementedError

    def recv(self) -> Tuple[str, bytes]:
        """
        Overview:
            Wait for incoming message, this function will block the current thread.
        Returns:
            - data (:obj:`Any`): The sent payload.
        """
        raise NotImplementedError

    def stop(self) -> None:
        """
        Overview:
            Unsubscribe from all topics and stop the connection to the message queue server.
        """
        return
