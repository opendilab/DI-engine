import torch
import psutil
import os
import functools
import warnings
import torch.profiler as torch_profiler

from collections import OrderedDict, defaultdict, namedtuple
from torch.autograd.profiler_util import _format_time, _format_memory
from typing import List
from torch.autograd.profiler_util import FunctionEvent
# import torch.autograd.profiler as torch_profiler
from torch.profiler import record_function

Trace = namedtuple("Trace", ["path", "leaf", "module"])
CUSTOM_CASE = {
    'do_dropout2_case1',
    'do_dropout2_case2',
    'do_dropout1_case1',
    'do_dropout1_case2',
    'do_dropout1_case3',
}


def walk_modules(module, name="", path=()):
    """Generator. Walks through a PyTorch Module and outputs Trace tuples"""
    if not name:
        name = module.__class__.__name__
    named_children = list(module.named_children())
    path = path + (name, )
    yield Trace(path, len(named_children) == 0, module)
    # recursively walk into all submodules
    for name, child_module in named_children:
        yield from walk_modules(child_module, name=name, path=path)


class LayerProfileV2(object):
    """Layer by layer profiling of PyTorch models, using the PyTorch autograd profiler.
    (WGT): If you run train_llm.py, please do not turn on 'record_shapes' and 'with_stack',
    it will lead to CUDA OOM.
    """

    def __init__(
        self,
        model,
        enabled=True,
        use_cuda=False,
        profile_memory=False,
        paths=None,
        with_flops=False,
        record_shapes=False,
        with_stack=False
    ):
        self._model = model
        self.enabled = enabled
        self.use_cuda = use_cuda
        self.profile_memory = profile_memory
        self.with_flops = with_flops
        self.record_shapes = record_shapes
        self.paths = paths
        self.with_stack = with_stack

        self.entered = False
        self.exited = False
        self.traces = ()
        self._ids = set()
        self.trace_profile_events = defaultdict(list)

    def __enter__(self):
        if not self.enabled:
            return self
        if self.entered:
            raise RuntimeError("torchprof profiler is not reentrant")
        self.entered = True
        self._forwards = {}  # store the original forward functions
        self.traces = tuple(map(self._hook_trace, walk_modules(self._model)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        tuple(map(self._remove_hook_trace, self.traces))
        del self._forwards  # remove unnecessary forwards
        self.exited = True
        print(self.display())

    def __str__(self):
        return self.display()

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def _hook_trace(self, trace):
        [path, leaf, module] = trace
        _id = id(module)
        name = path[-1]
        # 只有叶子节点会被拦截 forward
        # if (self.paths is not None and path in self.paths) or (
        #     self.paths is None and leaf
        # ):
        if leaf:
            if _id in self._ids:
                #already wrapped
                return trace
            self._ids.add(_id)

            global CUSTOM_CASE
            if name in CUSTOM_CASE:
                _forward = getattr(module, name)
                # _forward = module.dropout_add_layer_norm
            else:
                _forward = module.forward

            print(f"wrap: fn:{_forward}", flush=True)
            self._forwards[path] = _forward

            @functools.wraps(_forward)
            def wrap_forward(*args, **kwargs):
                res = None
                pre = psutil.Process(os.getpid()).memory_info().rss
                try:
                    with torch_profiler.profile(activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                    ], use_cuda=self.use_cuda, profile_memory=self.profile_memory, with_stack=self.with_stack,
                                                record_shapes=self.record_shapes) as prof:
                        res = _forward(*args, **kwargs)
                except TypeError:
                    assert RuntimeError(" new `torch.profile` is unsupported in torch < 1.6")

                # event_list = prof.function_events
                event_list = prof.profiler.function_events
                # PyTorch up until version 1.7 exposes this method. From PyTorch 1.8 onwards,
                # it is called via EventList._build_tree at the end of the context manager.
                if hasattr(event_list, "populate_cpu_children"):
                    event_list.populate_cpu_children()

                # for i, event in enumerate(event_list)
                # if hasattr(event_list, 'cpu_memory_usage'):
                #     event_list['cpu_memory_usage'] = pre
                # else:
                #     event_list.__setattr__('cpu_memory_usage', pre)
                print(f"name:{name}, now cpu_memory_usage : {pre / (1024**3):.4f} GB")
                event_list.append(FunctionEvent(id=0, name=0, thread=0, start_us=0, end_us=0, cpu_memory_usage=pre))

                # each profile call should be contained in its own list
                self.trace_profile_events[path].append(event_list)

                return res

            if name in CUSTOM_CASE:
                setattr(module, name, wrap_forward)
            else:
                module.forward = wrap_forward
        else:
            # 为了看出来host的内存开销，我们需要给非叶子节点也加一个 hook
            _forward = module.forward

            @functools.wraps(_forward)
            def wrap_forward(*args, **kwargs):
                res = None
                pre = psutil.Process(os.getpid()).memory_info().rss
                res = _forward(*args, **kwargs)
                # 无法统计峰值内存占用？
                # event_list = prof.function_events
                # event_list = prof.profiler.function_events
                # each profile call should be contained in its own list

                # self.trace_profile_events[path].append(event_list)
                return res

        #  module.forward = wrap_forward

        return trace

    def _remove_hook_trace(self, trace):
        [path, leaf, module] = trace
        _id = id(module)
        if _id in self._ids:
            self._ids.discard(_id)
        else:
            return
        if (self.paths is not None and path in self.paths) or (self.paths is None and leaf):
            module.forward = self._forwards[path]

    def raw(self):
        if self.exited:
            return (self.traces, self.trace_profile_events)

    def display_DFS(self, show_events=False):
        return traces_to_display_BFS(
            self.traces,
            self.trace_profile_events,
            show_events=show_events,
            paths=self.paths,
            use_cuda=self.use_cuda,
            profile_memory=self.profile_memory,
        )

    def display(self, show_events=False):
        if self.exited:
            return traces_to_display(
                self.traces,
                self.trace_profile_events,
                show_events=show_events,
                paths=self.paths,
                use_cuda=self.use_cuda,
                profile_memory=self.profile_memory,
            )
        return "<unfinished torchprof.profile>"


class MMeasure:

    def __init__(self) -> None:
        self.self_cpu_total = 0
        self.cpu_total = 0
        self.self_cuda_total = 0
        self.cuda_total = 0
        self.self_cpu_memory = 0
        self.cpu_memory = 0
        self.self_cuda_memory = 0
        self.cuda_memory = 0
        self.occurrences = 0


Measure = namedtuple(
    "Measure",
    [  # when attr value is None, profiler unsupported
        "self_cpu_total",
        "cpu_total",
        "self_cuda_total",
        "cuda_total",
        "self_cpu_memory",
        "cpu_memory",
        "self_cuda_memory",
        "cuda_memory",
        "occurrences",
    ],
)


def _flatten_tree(t, depth=0):
    flat = []
    for name, st in t.items():
        measures = st.pop(None, None)
        flat.append([depth, name, measures])
        flat.extend(_flatten_tree(st, depth=depth + 1))
    return flat


def _build_measure_tuple(events, occurrences):
    # Events may have missing attributes depending on the PyTorch version used.
    # memory profiling supported in torch >= 1.6
    events: List[FunctionEvent]
    self_cpu_memory = None
    has_self_cpu_memory = any(hasattr(e, "self_cpu_memory_usage") for e in events)
    if has_self_cpu_memory:
        # 因为只有一个 cpu mem的值，我们直接max就行
        self_cpu_memory = max([getattr(e, "self_cpu_memory_usage", 0) for e in events])
    cpu_memory = None
    has_cpu_memory = any(hasattr(e, "cpu_memory_usage") for e in events)
    if has_cpu_memory:
        cpu_memory = max([getattr(e, "cpu_memory_usage", 0) for e in events])
    self_cuda_memory = None
    has_self_cuda_memory = any(hasattr(e, "self_cuda_memory_usage") for e in events)
    if has_self_cuda_memory:
        self_cuda_memory = sum([getattr(e, "self_cuda_memory_usage", 0) for e in events])
    cuda_memory = None
    has_cuda_memory = any(hasattr(e, "cuda_memory_usage") for e in events)
    if has_cuda_memory:
        cuda_memory = sum([getattr(e, "cuda_memory_usage", 0) for e in events])

    # self CUDA time supported in torch >= 1.7
    self_cuda_total = None
    has_self_cuda_time = any(hasattr(e, "self_cuda_time_total") for e in events)
    if has_self_cuda_time:
        self_cuda_total = sum([getattr(e, "self_cuda_time_total", 0) for e in events])

    return Measure(
        self_cpu_total=sum([e.self_cpu_time_total for e in events]),
        cpu_total=sum([e.cpu_time_total for e in events]),
        self_cuda_total=self_cuda_total,
        cuda_total=sum([e.cuda_time_total for e in events]),
        self_cpu_memory=self_cpu_memory,
        cpu_memory=cpu_memory,
        self_cuda_memory=self_cuda_memory,
        cuda_memory=cuda_memory,
        occurrences=occurrences,
    )


def _format_measure_tuple(measure):
    format_memory = _format_memory

    self_cpu_total = (_format_time(measure.self_cpu_total) if measure else "")

    cpu_total = _format_time(measure.cpu_total) if measure else ""
    self_cuda_total = (_format_time(measure.self_cuda_total) if measure and measure.self_cuda_total is not None else "")

    cuda_total = _format_time(measure.cuda_total) if measure else ""
    self_cpu_memory = (
        format_memory(measure.self_cpu_memory) if measure and measure.self_cpu_memory is not None else ""
    )
    cpu_memory = (format_memory(measure.cpu_memory) if measure and measure.cpu_memory is not None else "")
    self_cuda_memory = (
        format_memory(measure.self_cuda_memory) if measure and measure.self_cuda_memory is not None else ""
    )
    cuda_memory = (format_memory(measure.cuda_memory) if measure and measure.cuda_memory is not None else "")
    occurrences = str(measure.occurrences) if measure else ""

    return Measure(
        self_cpu_total=self_cpu_total,
        cpu_total=cpu_total,
        self_cuda_total=self_cuda_total,
        cuda_total=cuda_total,
        self_cpu_memory=self_cpu_memory,
        cpu_memory=cpu_memory,
        self_cuda_memory=self_cuda_memory,
        cuda_memory=cuda_memory,
        occurrences=occurrences,
    )


def group_by(events, keyfn):
    event_groups = OrderedDict()
    for event in events:
        key = keyfn(event)
        key_events = event_groups.get(key, [])
        key_events.append(event)
        event_groups[key] = key_events
    return event_groups.items()


# 将拥有相同前缀的路径的结果合并为一起
def sum_prefix_time():
    pass


def add_measure(lhs, rhs):
    # lhs.self_cpu_total +=  rhs.self_cpu_total if hasattr(rhs, 'self_cpu_total') else 0
    # lhs.cpu_total += rhs.cpu_total if hasattr(rhs, 'cpu_total') else 0
    # lhs.self_cuda_total += rhs.self_cuda_total if hasattr(rhs, 'self_cuda_total') else 0
    # lhs.cuda_total += rhs.cuda_total if hasattr(rhs, 'cuda_total') else 0
    # lhs.self_cpu_memory += rhs.self_cpu_memory if hasattr(rhs, 'self_cpu_memory') else 0
    # lhs.cpu_memory += rhs.cpu_memory if hasattr(rhs, 'cpu_memory') else 0
    # lhs.self_cuda_memory += rhs.self_cuda_memory if hasattr(rhs, 'self_cuda_memory') else 0
    # lhs.cuda_memory += rhs.cuda_memory if hasattr(rhs, 'cuda_memory') else 0
    # lhs.occurrences += rhs.occurrences if hasattr(rhs, 'occurrences') else 0
    lhs.self_cpu_total += rhs.self_cpu_total if rhs.self_cpu_total is not None else 0
    lhs.cpu_total += rhs.cpu_total if rhs.cpu_total is not None else 0
    lhs.self_cuda_total += rhs.self_cuda_total if rhs.self_cuda_total is not None else 0
    lhs.cuda_total += rhs.cuda_total if rhs.cuda_total is not None else 0
    # lhs.self_cpu_memory += rhs.self_cpu_memory if rhs.self_cpu_memory is not None else 0
    # lhs.cpu_memory += rhs.cpu_memory  if rhs.cpu_memory is not None else 0
    lhs.self_cuda_memory += rhs.self_cuda_memory if rhs.self_cuda_memory is not None else 0
    lhs.cuda_memory += rhs.cuda_memory if rhs.cuda_memory is not None else 0
    lhs.occurrences += rhs.occurrences if rhs.occurrences is not None else 0


# tree is not change
# trace_events is not change
def BFS(trace_events, traces, tree, width, path_idx, paths=None, show_events=False):
    [path, leaf, module] = traces[width]
    # name = path[-1]
    name = "/"
    for sub_name in path:
        name = os.path.join(name, sub_name)
    print(f"name: {name}, tree key :{tree.keys()}")
    assert name not in tree

    tree[name] = OrderedDict()
    current_tree = tree[name]
    if leaf:
        print(f"BFS: name:{name}, {path}, {leaf} is leaf")
        events = [te for t_events in trace_events[path] for te in t_events]
        if show_events:
            for event_name, event_group in group_by(events, lambda e: e.name):
                event_group = list(event_group)
                current_tree[event_name] = {None: _build_measure_tuple(event_group, len(event_group))}
        else:
            current_tree[None] = _build_measure_tuple(events, len(trace_events[path]))
    else:
        print(f"BFS: name:{name}, {path}, {leaf} is non-leaf")
        for child_index in range(width + 1, len(traces)):
            [path, leaf, module] = traces[child_index]
            if len(list(path)) > path_idx + 2:
                continue
            # 如果前缀一样，说明是当前节点的儿子节点
            sub_path = "/"
            for i, sub_name in enumerate(path):
                if i > path_idx:
                    break
                sub_path = os.path.join(sub_path, sub_name)
            # sub_path = path.split('/')[0:path_idx+1]
            # if name == path[path_idx]:
            if name == sub_path:
                print(f"BFS: name:{name} ready to enter son :{path}")
                BFS(trace_events, traces, current_tree, child_index, path_idx + 1)
            else:
                break

        # 总结返回的结果
        my_measure = MMeasure()
        for son_name, odict in current_tree.items():
            print(f"{name} build {son_name}, {odict[None]}")
            add_measure(my_measure, odict[None])

        if show_events:
            assert False
        else:
            current_tree[None] = Measure(
                self_cpu_total=my_measure.self_cpu_total,
                cpu_total=my_measure.cpu_total,
                self_cuda_total=my_measure.self_cuda_total,
                cuda_total=my_measure.cuda_total,
                self_cpu_memory=my_measure.self_cpu_memory,
                cpu_memory=my_measure.cpu_memory,
                self_cuda_memory=my_measure.self_cuda_memory,
                cuda_memory=my_measure.cuda_memory,
                occurrences=my_measure.occurrences
            )
            print(f"finish build :{current_tree[None]}")

    return tree


def DFS(traces, tree, paths, show_events, trace_events):
    print(f"len:{len(traces)}")
    for i, trace in enumerate(traces):
        print(type(trace), i, trace)
        [path, leaf, module] = trace
        current_tree = tree
        # unwrap all of the events, in case model is called multiple times
        events = [te for t_events in trace_events[path] for te in t_events]
        # 深度优先遍历？
        for depth, name in enumerate(path, 1):
            if name not in current_tree:
                current_tree[name] = OrderedDict()
            # if depth == len(path) and (
            #     (paths is None and leaf) or (paths is not None and path in paths)
            # ):
            # 只有叶子节点可以 measure，但是我们现在让非叶子节点也参与进来
            # 因为非叶子节点没有做 profile，只能通过累加叶子节点的和做到
            if depth == len(path):
                if (paths is None and leaf) or (paths is not None and path in paths):
                    # tree measurements have key None, avoiding name conflict
                    if show_events:
                        for event_name, event_group in group_by(events, lambda e: e.name):
                            event_group = list(event_group)
                            current_tree[name][event_name] = {None: _build_measure_tuple(event_group, len(event_group))}
                    else:
                        current_tree[name][None] = _build_measure_tuple(events, len(trace_events[path]))
            current_tree = current_tree[name]
    return tree


def traces_to_display_BFS(
    traces,
    trace_events,
    show_events=False,
    paths=None,
    use_cuda=False,
    profile_memory=False,
    # dt = ('|', '|-- ', '+-- ', ' ') # ascii
    dt=("\u2502", "\u251c\u2500\u2500 ", "\u2514\u2500\u2500 ", " "),  # ascii-ex
):
    tree = OrderedDict()
    # import torch.distributed as dist
    # print("show trace___________")
    # print(traces)
    tree = BFS(trace_events, traces, tree, 0, 0)


def traces_to_display(
    traces,
    trace_events,
    show_events=False,
    paths=None,
    use_cuda=False,
    profile_memory=False,
    # dt = ('|', '|-- ', '+-- ', ' ') # ascii
    dt=("\u2502", "\u251c\u2500\u2500 ", "\u2514\u2500\u2500 ", " "),  # ascii-ex
):
    """Construct human readable output of the profiler traces and events."""
    tree = OrderedDict()
    print("show trace___________")
    print(traces)

    # tree = DFS(traces, tree, paths, show_events, trace_events)
    # tree = DFS(trace_events, traces, tree, 0, 0)
    tree = BFS(trace_events, traces, tree, 0, 0)
    tree_lines = _flatten_tree(tree)

    format_lines = []
    has_self_cuda_total = False
    has_self_cpu_memory = False
    has_cpu_memory = False
    has_self_cuda_memory = False
    has_cuda_memory = False

    for idx, tree_line in enumerate(tree_lines):
        depth, name, measures = tree_line

        next_depths = [pl[0] for pl in tree_lines[idx + 1:]]
        pre = ""
        if depth > 0:
            pre = dt[1] if depth in next_depths and next_depths[0] >= depth else dt[2]
            depth -= 1
        while depth > 0:
            pre = (dt[0] + pre) if depth in next_depths else (dt[3] + pre)
            depth -= 1
        format_lines.append([pre + name, *_format_measure_tuple(measures)])
        if measures:
            has_self_cuda_total = (has_self_cuda_total or measures.self_cuda_total is not None)
            has_self_cpu_memory = (has_self_cpu_memory or measures.self_cpu_memory is not None)
            has_cpu_memory = has_cpu_memory or measures.cpu_memory is not None
            has_self_cuda_memory = (has_self_cuda_memory or measures.self_cuda_memory is not None)
            has_cuda_memory = has_cuda_memory or measures.cuda_memory is not None

    # construct the table (this is pretty ugly and can probably be optimized)
    heading = (
        "Module",
        "Self CPU total",
        "CPU total",
        "Self CUDA total",
        "CUDA total",
        "Self CPU Mem",
        "CPU Mem",
        "Self CUDA Mem",
        "CUDA Mem",
        "Number of Calls",
    )
    max_lens = [max(map(len, col)) for col in zip(*([heading] + format_lines))]

    # not all columns should be displayed, specify kept indexes
    keep_indexes = [0, 1, 2, 9]
    if profile_memory:
        if has_self_cpu_memory:
            keep_indexes.append(5)
        if has_cpu_memory:
            keep_indexes.append(6)
    if use_cuda:
        if has_self_cuda_total:
            keep_indexes.append(3)
        keep_indexes.append(4)
        if profile_memory:
            if has_self_cuda_memory:
                keep_indexes.append(7)
            if has_cuda_memory:
                keep_indexes.append(8)
    keep_indexes = tuple(sorted(keep_indexes))

    display = (  # table heading
        " | ".join(
            [
                "{:<{}s}".format(heading[keep_index], max_lens[keep_index])
                for keep_index in keep_indexes
            ]
        )
        + "\n"
    )
    display += (  # separator
        "-|-".join(
            [
                "-" * max_len
                for val_idx, max_len in enumerate(max_lens)
                if val_idx in keep_indexes
            ]
        )
        + "\n"
    )
    for format_line in format_lines:  # body
        display += (
            " | ".join(
                [
                    "{:<{}s}".format(value, max_lens[val_idx])
                    for val_idx, value in enumerate(format_line) if val_idx in keep_indexes
                ]
            ) + "\n"
        )

    return display
