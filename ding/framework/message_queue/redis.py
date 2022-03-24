from ding.framework.message_queue.mq import MQ
from ding.utils import MQ_REGISTRY


@MQ_REGISTRY.register("redis")
class RedisMQ(MQ):
    """
    Overview:
        Connect distributed processes with redis
    """
    pass
