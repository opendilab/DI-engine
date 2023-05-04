import torch
from enum import Enum
from ding.utils.prof.sub_profiler.comm_profiler import v2CommProfiler
from ding.utils.prof.sub_profiler.layer_profiler import LayerProfile
from ding.utils.prof.sub_profiler.layer_profiler_v2 import LayerProfileV2
from ding.utils.prof.sub_profiler.mem_profiler import MemHook
from ding.utils.prof.sub_profiler.sys_metric_tracker import MetricTracker
# import torch.autograd.profiler


class DummyProfile:

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, a, b, c):
        pass

    def step(self):
        pass


class ProfileType(Enum):
    DUMMY = 'Dummy'
    COMM = 'Comm'
    LAYER = 'Layer'
    MEM = 'MEM'
    TORCH = 'Torch'


LAYER_PRODILER_CONFIG = {}

# NAME_DICT = {ProfileType.DUMMY: DummyProfile,
#              ProfileType.COMM: v2CommProfiler,
#              ProfileType.LAYER: LayerProfile,
#              ProfileType.MEM: MemHook,
#              ProfileType.TORCH: torch.profiler.profile}


# TODO: Considering the problem that the old version of pytorch cannot use the new profile interface.
def get_profiler(enable: ProfileType, trace_path=None):
    if enable == ProfileType.DUMMY:
        profiler = DummyProfile
    elif enable == ProfileType.TORCH:
        assert trace_path is not None
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(skip_first=5, wait=1, warmup=1, active=1, repeat=0),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{trace_path}"),
            with_stack=True,
            with_modules=True
        )
    elif enable == ProfileType.COMM:
        profiler = v2CommProfiler(enable_profile=True, fire_step=3)
    elif enable == ProfileType.LAYER:
        profiler = LayerProfileV2
        # profiler = LayerProfile
    elif enable == ProfileType.MEM:
        profiler = MemHook

    return profiler
