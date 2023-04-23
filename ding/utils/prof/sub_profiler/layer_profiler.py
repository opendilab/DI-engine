# MIT License

# Copyright (c) 2019 Alexander W. Wong

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:

# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
# OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
# OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
import torch.profiler as torch_profiler
# import torch.autograd.profiler as torch_profiler

import functools
import warnings
from collections import OrderedDict, namedtuple
from torch.autograd.profiler_util import _format_time, _format_memory
from typing import List
from torch.autograd.profiler_util import FunctionEvent
from collections import OrderedDict, defaultdict, namedtuple

Trace = namedtuple("Trace", ["path", "leaf", "module"])
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


class LayerProfile(object):
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

    def __str__(self):
        return self.display()

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def _hook_trace(self, trace):
        [path, leaf, module] = trace
        _id = id(module)
        if (self.paths is not None and path in self.paths) or (self.paths is None and leaf):
            if _id in self._ids:
                #already wrapped
                return trace
            self._ids.add(_id)
            _forward = module.forward
            self._forwards[path] = _forward

            @functools.wraps(_forward)
            def wrap_forward(*args, **kwargs):
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
                # each profile call should be contained in its own list
                self.trace_profile_events[path].append(event_list)
                return res

            module.forward = wrap_forward
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
        self_cpu_memory = sum([getattr(e, "self_cpu_memory_usage", 0) for e in events])
    cpu_memory = None
    has_cpu_memory = any(hasattr(e, "cpu_memory_usage") for e in events)
    if has_cpu_memory:
        cpu_memory = sum([getattr(e, "cpu_memory_usage", 0) for e in events])
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

    for trace in traces:
        [path, leaf, module] = trace
        current_tree = tree
        # unwrap all of the events, in case model is called multiple times
        events = [te for t_events in trace_events[path] for te in t_events]
        for depth, name in enumerate(path, 1):
            if name not in current_tree:
                current_tree[name] = OrderedDict()
            if depth == len(path) and ((paths is None and leaf) or (paths is not None and path in paths)):
                # tree measurements have key None, avoiding name conflict
                if show_events:
                    for event_name, event_group in group_by(events, lambda e: e.name):
                        event_group = list(event_group)
                        current_tree[name][event_name] = {None: _build_measure_tuple(event_group, len(event_group))}
                else:
                    current_tree[name][None] = _build_measure_tuple(events, len(trace_events[path]))
            current_tree = current_tree[name]
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
