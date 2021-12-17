from tensorboardX import SummaryWriter
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ding.framework import Task


class DistributedWriter(SummaryWriter):

    def __init__(self, *args, **kwargs):
        self._default_writer_to_disk = kwargs.get("write_to_disk") if "write_to_disk" in kwargs else True
        kwargs["write_to_disk"] = False
        super().__init__(*args, **kwargs)
        self._in_parallel = False
        self._task = None
        self._is_writer = False
        self._lazy_initialized = False

    def plugin(self, task: "Task", is_writer: False) -> "DistributedWriter":
        # TODO add buffer
        if task.router.is_active:
            self._in_parallel = True
            self._task = task
            self._is_writer = is_writer
            if is_writer:
                self.initialize()
            self._lazy_initialized = True
            task.router.register_rpc("distributed_writer", self.on_distributed_writer)
            task.once("exit", lambda: self.close())
        return self

    def on_distributed_writer(self, fn_name: str, *args, **kwargs):
        if self._is_writer:
            getattr(self, fn_name)(*args, **kwargs)

    def initialize(self):
        self.close()
        self._write_to_disk = self._default_writer_to_disk
        self._get_file_writer()
        self._lazy_initialized = True

    def __del__(self):
        self.close()


def enable_parallel(fn_name, fn):

    def _parallel_fn(self: DistributedWriter, *args, **kwargs):
        if not self._lazy_initialized:
            self.initialize()
        if self._in_parallel and not self._is_writer:
            self._task.router.send_rpc("distributed_writer", fn_name, *args, **kwargs)
        else:
            fn(self, *args, **kwargs)

    return _parallel_fn


ready_to_parallel_fns = [
    "add_scalar",
]
for fn_name in ready_to_parallel_fns:
    setattr(DistributedWriter, fn_name, enable_parallel(fn_name, getattr(DistributedWriter, fn_name)))
