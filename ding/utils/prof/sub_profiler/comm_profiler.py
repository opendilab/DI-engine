import inspect
import time
import torch
import torch.distributed as dist
from torch.distributed import get_backend, ReduceOp
from functools import partial
from typing import List, Optional
from ding.utils.prof.sub_profiler.utils import _format_memory, _format_time, _format_bandwidth

# collective comm
torch_all_reduce = dist.all_reduce
torch_broadcast = dist.broadcast
torch_reduce = dist.reduce
torch_all_gather = dist.all_gather
# torch_all_gather_into_tensor = dist.all_gather_into_tensor
# torch_all_gather_coalesced = dist.all_gather_coalesced
# torch_reduce_scatter = dist.reduce_scatter


class DummyRecordFunction:
    """
    DummyRecordFunction
    """

    def __init__(self, name, logger=None) -> None:
        self.time = 0
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.time = time.time()
        return self

    def __exit__(self, a, b, c):
        # if is_dp_rank_0() and is_tp_rank_0():
        self.logger.info(f"{self.name} use time: {time.time() - self.time} s", ranks=[0])


# Find longest prefix substr of given strings.
def longestCommonPrefix(strs: List[str]):
    res = ""
    for tmp in zip(*strs):
        tmp_set = set(tmp)
        if len(tmp_set) == 1:
            res += tmp[0]
        else:
            break
    return res


def _get_code_location(depth: int):
    ret = []
    length = min(len(inspect.stack()), depth + 1)
    for i in range(3, length):
        upper_frame = inspect.stack()[i]
        function_name = inspect.stack()[i - 1].function
        ret.append(upper_frame.filename)
        ret.append("(")
        ret.append(str(upper_frame.lineno))
        ret.append("): ")
        ret.append(function_name)
        if i != length - 1:
            ret.append("\n")

    return "".join(ret)


class CommEvent:
    """Communication Event. Used for communication time and communication
    volume recording.
    """

    def __init__(self):
        self.self_count = 0
        self.self_comm_vol = 0.0
        self.self_cuda_time = 0.0
        self.max_self_cuda_time = 0
        self.min_self_cuda_time = float("inf")

        self.copy_cuda_time = 0.0
        self.h2d_time = 0.0
        self.h2d_count = 0
        self.d2h_time = 0.0
        self.d2h_count = 0

    def add(self, rhs):
        self.self_count += rhs.self_count
        self.self_comm_vol += rhs.self_comm_vol
        self.self_cuda_time += rhs.self_cuda_time

        self.max_self_cuda_time = max(self.max_self_cuda_time, rhs.self_cuda_time)
        self.min_self_cuda_time = min(self.min_self_cuda_time, rhs.self_cuda_time)

        self.copy_cuda_time = rhs.copy_cuda_time
        self.h2d_count = rhs.h2d_count
        self.h2d_time = rhs.h2d_time
        self.d2h_count = rhs.d2h_count
        self.d2h_time = rhs.d2h_time

    def __str__(self) -> str:
        return f"self_count:{self.self_count}, self_comm_vol:{self.self_comm_vol}, \
self_cuda_time:{self.self_cuda_time}, copy_cuda_time:{self.copy_cuda_time}"


class profile_data:
    """
    profile_data
    """

    def __init__(self, name=None, stacks=None, dur=0, sendvol=0, recvvol=0, shapes=[]) -> None:  # pylint: disable=W0102
        self.name = name
        self.stacks = stacks
        self.dur = dur
        self.sendvol = sendvol
        self.recvvol = recvvol
        self.shapes = shapes

    def __str__(self) -> str:
        return f"{self.name}\n recv.vol:{_format_memory(self.recvvol)}, send.vol:{_format_memory(self.sendvol)}, \
dtime:{_format_time(self.dur)}"


class v2CommProfiler:
    """Communication profiler. Records all communication events."""

    PCIE_KN_SET = {"Memcpy HtoD", "Memcpy DtoH", "aten::copy_"}

    def __init__(
        self,
        depth: int = 0,
        enable_profile=False,
        enable_p2p=True,
        enable_coll=True,
        fire_step=-1,
        id_str="",
        **kwargs
    ):
        self.depth = 3 + depth
        self.enable_profile = enable_profile
        self.enable_p2p = enable_p2p
        self.enable_coll = enable_coll
        self.fire_step = fire_step
        self.now_step = 0
        self.comm_profile_map = {}
        self.id_str = id_str
        self.ops_record = dict()
        self.warn_flag = False

    def reset(self):
        self.ops_record.clear()

    def __enter__(self):
        return self

    def fire(self):
        print("profiler __fire__!", flush=True)
        if self.enable_profile:
            if self.enable_p2p:
                pass
            if self.enable_coll:
                dist.all_reduce = partial(all_reduce, profiler=self)
                dist.broadcast = partial(broadcast, profiler=self)

    def cool(self):
        print("profiler cool!", flush=True)
        if self.enable_profile:
            if self.enable_p2p:
                pass
            if self.enable_coll:
                dist.all_reduce = torch_all_reduce
                dist.broadcast = torch_broadcast

    def step(self):
        self.now_step += 1
        if self.fire_step == self.now_step:
            self.fire()
        else:
            self.cool()
            print(f"fire_step:{self.fire_step}, self.now_step: {self.now_step}")
            if self.fire_step + 1 == self.now_step:
                print("show!")
                self.show()

    def __exit__(self, exc_type, exc_val, exc_tb):  # pylint: disable=E0302
        if self.fire_step != self.now_step:
            pass

    def show(self):
        if self.enable_profile:
            for _, v in self.comm_profile_map.items():
                print(v)

            print(f"{self.result_str()}", flush=True)
            print(self.result_str(), 0)

    def result_str(self, sep: str = "\n"):
        res = []

        def append(s: str = None):
            if s is not None:
                res.append(s)
            res.append(sep)

        if self.warn_flag:
            append(
                "Warnning: there exists multiple communication operations in the same time. As a result, "
                "the profiling result is not accurate."
            )

        append("Collective communication profiling result:")
        append("All events:")

        seperation = '-' * 74
        row_format = '{:^10}' + '{:^12}' * 2 + '{:^16}' + '{:^12}' * 3

        append(seperation)
        append(
            row_format.format(
                'Location', 'GPU time', 'Percentage', 'Comm volume', 'Bandwidth', 'PCIe BW', 'Num of calls'
            )
        )
        append(seperation)

        show_list = sorted(self.ops_record.items(), key=lambda kv: -kv[1].self_cuda_time)
        for location, event in show_list:
            event: CommEvent
            append(location)
            append(
                row_format.format(
                    '',
                    _format_time(event.self_cuda_time),
                    # '{:.1f}%'.format(event.self_cuda_time / self.total_cuda_time * 100.0),
                    '{:.1f}%'.format(0.0),
                    _format_memory(event.self_comm_vol),
                    _format_bandwidth(event.self_comm_vol, event.self_cuda_time),
                    _format_bandwidth(event.self_comm_vol, event.copy_cuda_time),
                    event.self_count
                )
            )
            append()

        return ''.join(res)

    def to_tensorboard(self):
        pass

    def to_file(self):
        pass

    def add_info(self, name, sendvol, recvvol, dur):
        if name not in self.comm_profile_map:
            self.comm_profile_map.update(
                {name: profile_data(name=self.id_str + name, dur=dur, sendvol=sendvol, recvvol=recvvol)}
            )
        else:
            self.comm_profile_map[name].sendvol += sendvol
            self.comm_profile_map[name].recvvol += recvvol
            self.comm_profile_map[name].dur += dur

    def add_record(self, prefix, event: CommEvent):
        if event.self_count > 0:
            if prefix in self.ops_record:
                self.ops_record[prefix].add(event)
            else:
                self.ops_record[prefix] = event


def get_comm_col(tt):
    vol = 0
    shape_list = []
    if isinstance(tt, (List, tuple)):
        for item in tt:
            if isinstance(item, torch.Tensor):
                shape_list.append(item.shape)
                vol += item.element_size() * item.numel()
    elif isinstance(tt, torch.Tensor):
        vol += tt.element_size() * tt.numel()
        shape_list.append(tt.shape)
    else:
        raise RuntimeError(f"get_comm_col get type:{type(tt)}")

    return vol, shape_list


def all_reduce(
    tensor: torch.Tensor,
    op: ReduceOp = ReduceOp.SUM,
    group=None,
    async_op: bool = False,
    profiler: v2CommProfiler = None
):
    # async_check(profiler)
    comm_size = dist.get_world_size(group)
    correction = 2 * (comm_size - 1) / comm_size
    comm_vol = correction * tensor.element_size() * tensor.numel()

    if async_op:
        print(f"{profiler.id_str}: skip async all-reduce, comm_vol: {comm_vol}")
        return torch_all_reduce(tensor, op, group, async_op)

    s = time.time()
    torch_all_reduce(tensor, op, group, async_op)
    torch.cuda.synchronize()
    total_cuda_time = (time.time() - s) * (1000 ** 2)  # change it to us
    curr_event = CommEvent()
    curr_event.self_count = 1
    curr_event.self_comm_vol = comm_vol
    curr_event.self_cuda_time = total_cuda_time
    profiler.add_record(_get_code_location(profiler.depth), curr_event)
    return None


def broadcast(tensor: torch.Tensor, src: int, group=None, async_op: bool = False, profiler: v2CommProfiler = None):

    comm_vol = 1.0 * tensor.element_size() * tensor.numel()

    if async_op:
        print(f"{profiler.id_str}: skip async broadcast, comm_vol: {comm_vol}")
        return torch_broadcast(tensor, src, group, async_op)

    s = time.time()
    torch_broadcast(tensor, src, group, async_op)
    torch.cuda.synchronize()
    total_cuda_time = (time.time() - s) * (1000 ** 2)  # change it to us
    curr_event = CommEvent()
    curr_event.self_count = 1
    curr_event.self_comm_vol = comm_vol
    curr_event.self_cuda_time = total_cuda_time
    profiler.add_record(_get_code_location(profiler.depth), curr_event)
    return None


def reduce(
    tensor: torch.Tensor,
    dst: int,
    op: ReduceOp = ReduceOp.SUM,
    group=None,
    async_op: bool = False,
    profiler: v2CommProfiler = None
):

    if async_op:
        print(f"{profiler.id_str}: skip async reduce, comm_vol: {comm_vol}")
        return torch_reduce(tensor, dst, op, group, async_op)

    comm_vol = 1.0 * tensor.element_size() * tensor.numel()
    s = time.time()
    torch_reduce(tensor, dst, op, group, async_op)
    total_cuda_time = (time.time() - s) * (1000 ** 2)  # change it to us
    curr_event = CommEvent()
    curr_event.self_count = 1
    curr_event.self_comm_vol = comm_vol
    curr_event.self_cuda_time = total_cuda_time
    profiler.add_record(_get_code_location(profiler.depth), curr_event)
    return None


def all_gather(
    tensor_list: List[torch.Tensor],
    tensor: torch.Tensor,
    group=None,
    async_op: bool = False,
    profiler: v2CommProfiler = None
):

    if async_op:
        print(f"{profiler.id_str}: skip async all_gather, comm_vol: {comm_vol}")
        return torch_all_gather(tensor_list, tensor, group, async_op)
    comm_size = dist.get_world_size(group)
    correction = (comm_size - 1) / comm_size
    comm_vol = 0
    for ten in tensor_list:
        comm_vol += ten.element_size() * ten.numel()
    comm_vol *= correction
    s = time.time()
    torch_all_gather(tensor_list, tensor, group, async_op)
    total_cuda_time = (time.time() - s) * (1000 ** 2)  # change it to us
    curr_event = CommEvent()
    curr_event.self_count = 1
    curr_event.self_comm_vol = comm_vol
    curr_event.self_cuda_time = total_cuda_time
    profiler.add_record(_get_code_location(profiler.depth), curr_event)
    return None
