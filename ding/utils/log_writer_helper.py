from typing import TYPE_CHECKING

from tensorboardX import SummaryWriter

if TYPE_CHECKING:
    # TYPE_CHECKING is always False at runtime, but mypy will evaluate the contents of this block.
    # So if you import this module within TYPE_CHECKING, you will get code hints and other benefits.
    # Here is a good answer on stackoverflow:
    # https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports
    from ding.framework import Parallel


class DistributedWriter(SummaryWriter):
    """
    Overview:
        A simple subclass of SummaryWriter that supports writing to one process in multi-process mode.
        The best way is to use it in conjunction with the ``router`` to take advantage of the message \
            and event components of the router (see ``writer.plugin``).
    """

    def __init__(self, *args, **kwargs):
        self._default_writer_to_disk = kwargs.get("write_to_disk") if "write_to_disk" in kwargs else True
        # We need to write data to files lazily, so we should not use file writer in __init__,
        # On the contrary, we will initialize the file writer when the user calls the
        # add_* function for the first time
        kwargs["write_to_disk"] = False
        super().__init__(*args, **kwargs)
        self._in_parallel = False
        self._router = None
        self._is_writer = False
        self._lazy_initialized = False

    def plugin(self, router: "Parallel", is_writer: bool = False) -> "DistributedWriter":
        """
        Overview:
            Plugin ``router``, so when using this writer with active router, it will automatically send requests\
                to the main writer instead of writing it to the disk. So we can collect data from multiple processes\
                and write them into one file.
        Examples:
            >>> DistributedWriter().plugin(router, is_writer=True)
        """
        if router.is_active:
            self._in_parallel = True
            self._router = router
            self._is_writer = is_writer
            if is_writer:
                self.initialize()
            self._lazy_initialized = True
            router.on("distributed_writer", self._on_distributed_writer)
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
            self._router.emit("distributed_writer", fn_name, *args, **kwargs)
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
    if hasattr(DistributedWriter, fn_name):
        setattr(DistributedWriter, fn_name, enable_parallel(fn_name, getattr(DistributedWriter, fn_name)))
