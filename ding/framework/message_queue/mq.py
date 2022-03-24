from typing import Any


class MQ:
    """
    Overview:
        Abstract basic mq class.
    """

    def listen(self) -> None:
        """
        Overview:
            Bind to local socket or connect to third party components.
        """
        raise NotImplementedError

    def publish(self, topic: str, data: Any) -> None:
        """
        Overview:
            Send data to mq.
        Arguments:
            - topic (:obj:`str`): Topic.
            - data (:obj:`Any`): Payload data.
        """
        raise NotImplementedError

    def subscribe(self, topic: str) -> None:
        """
        Overview:
            Subscribe to the topic.
        Arguments:
            - topic (:obj:`str`): Topic
        """
        raise NotImplementedError

    def recv(self) -> Any:
        """
        Overview:
            Wait for incoming message, this function will block the current thread.
        Returns:
            - data (:obj:`Any`): The sent payload.
        """
        raise NotImplementedError
