from tensorboardX import SummaryWriter
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ding.framework import Task


class DistributedWriter(SummaryWriter):
    """
    Overview:
        A simple subclass of SummaryWriter that supports writing to one process in multi-process mode.
        The best way is to use it in conjunction with the ``task`` to take advantage of the message \
            and event components of the task (see ``writer.plugin``).
    """

    def __init__(self, *args, **kwargs):
        self._default_writer_to_disk = kwargs.get("write_to_disk") if "write_to_disk" in kwargs else True
        kwargs["write_to_disk"] = False
        super().__init__(*args, **kwargs)
        self._in_parallel = False
        self._task = None
        self._is_writer = False
        self._lazy_initialized = False

    def plugin(self, task: "Task", is_writer: False) -> "DistributedWriter":
        """
        Overview:
            Plugin ``task``, so when using this writer in the task pipeline, it will automatically send requests\
                to the main writer instead of writing it to the disk. So we can collect data from multiple processes\
                and write them into one file.
        Usage:
            ``DistributedWriter().plugin(task, is_writer=("node.0" in task.labels))``
        """
        if task.router.is_active:
            self._in_parallel = True
            self._task = task
            self._is_writer = is_writer
            if is_writer:
                self.initialize()
            self._lazy_initialized = True
            task.router.register_rpc("distributed_writer", self._on_distributed_writer)
            task.once("exit", lambda: self.close())
        return self

    def _on_distributed_writer(self, fn_name: str, *args, **kwargs):
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
    'add_audio',
    'add_custom_scalars',
    'add_custom_scalars_marginchart',
    'add_custom_scalars_multilinechart',
    'add_embedding',
    'add_figure',
    'add_graph',
    'add_graph_deprecated',
    'add_histogram',
    'add_histogram_raw',
    'add_hparams',
    'add_image',
    'add_image_with_boxes',
    'add_images',
    'add_mesh',
    'add_onnx_graph',
    'add_openvino_graph',
    'add_pr_curve',
    'add_pr_curve_raw',
    'add_scalar',
    'add_scalars',
    'add_text',
    'add_video',
]
for fn_name in ready_to_parallel_fns:
    setattr(DistributedWriter, fn_name, enable_parallel(fn_name, getattr(DistributedWriter, fn_name)))
