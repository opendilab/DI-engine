from .context import Context, OnlineRLContext, OfflineRLContext
from .task import Task, task, VoidMiddleware, enable_async
from .parallel import Parallel, MQType
from .event_loop import EventLoop
from .supervisor import Supervisor
from easydict import EasyDict
from ding.utils import DistributedWriter


def ding_init(cfg: EasyDict):
    DistributedWriter.get_instance(cfg.exp_name)
