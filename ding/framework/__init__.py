from .context import Context, OnlineRLContext, OfflineRLContext
from .task import Task, task
from .parallel import Parallel
from .event_loop import EventLoop
from .supervisor import Supervisor
from easydict import EasyDict
from ding.utils import DistributedWriter


def ding_init(cfg: EasyDict):
    DistributedWriter.get_instance(cfg.exp_name)
