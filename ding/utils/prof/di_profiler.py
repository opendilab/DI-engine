
import torch
# import torch.autograd.profiler

class DummyProfile:
    def __init__(self, *args, **kwargs) -> None:
        pass
    def __enter__(self):
        return self
    def __exit__(self, a, b, c):
        pass
    def step(self):
        pass

# TODO: Considering the problem that the old version of pytorch cannot use the new profile interface.
def init_profiler(enable = False, trace_path = None):
    if enable:
        assert trace_path is not None
        profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(skip_first=5, wait=1, warmup=1, active=1, repeat=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{trace_path}"),
                with_stack=True,
                with_modules=True
        ) 
    else:
        profiler = DummyProfile
    return profiler

