from typing import Any, Dict, List, Tuple
from collections import OrderedDict
from functools import partial, reduce

import os
import time
import torch
try:
    import pyecharts
except ImportError:
    import sys
    import logging
    logging.error("Please install pyecharts first, you can install it by running 'pip install pyecharts'")
    sys.exit(1)

MegaByte = 1024 * 1024


class SimpleMemState:
    """
    Overview:
        A class to represent the memory state of a model layer.
    Properties:
        ``layer_mem``, ``total_mem``
    Interfaces:
        ``add``, ``delete``, ``update_total_memory``, ``find_layer_state``, ``dump``, ``to_json``
    """

    def __init__(self, layer_name: str, layer_mem: int = 0) -> None:
        """
        Overview:
            Initialize the memory state of a model/tensors with the specific name.
        Arguments:
            - layer_name (:obj:`str`): The name of the layer.
            - layer_mem (:obj:`int`, optional): The memory usage of the layer in bytes. Defaults to 0.
        """
        self.layer_name = layer_name

        # Memory status of the current model layer.
        self._layer_mem: int = layer_mem
        # Total memory status of the model and sub-models, initialized with layer memory.
        self._total_mem: int = self._layer_mem
        # SimpleMemState of sub-models.
        self.sub_model_stats = OrderedDict()

    @property
    def layer_mem(self) -> int:
        """
        Overview:
            Get the memory usage of the layer.

        Returns:
            - layer_mem (:obj:`int`): The memory usage of the layer in bytes.
        """
        return self._layer_mem

    @layer_mem.setter
    def layer_mem(self, new_layer_mem: int) -> None:
        """
        Overview:
            Set the memory usage of the layer and update the total memory.
        Arguments:
            - new_layer_mem (:obj:`int`): The new memory usage of the layer in bytes.
        """
        diff = new_layer_mem - self._layer_mem
        self._layer_mem = new_layer_mem
        self._total_mem += diff

    @property
    def total_mem(self) -> int:
        """
        Overview:
            Get the total memory usage of the model and sub-models.

        Returns:
            - total_mem (:obj:`int`): The total memory usage of the model and sub-models in bytes.
        """
        return self._total_mem

    def add(self, layer_name: str, layer_mem: int = 0, flush: bool = True) -> None:
        """
        Overview:
            Add a layer to the memory state.
        Arguments:
            - layer_name (:obj:`str`): The name of the layer.
            - layer_mem (:obj:`int`, optional): The memory usage of the layer in bytes. Defaults to 0.
            - flush (:obj:`Optional[bool]`): Whether to update the total memory usage. Defaults to True.
        """
        path = layer_name.split(".")

        target = self.find_layer_state(path, create=True)
        target.layer_mem = layer_mem

        if flush:
            self.update_total_memory()

    def delete(self, layer_name: str, flush: bool = True) -> None:
        """
        Overview:
            Delete a layer from the memory state.
        Arguments:
            - layer_name (:obj:`str`): The name of the layer.
            - flush (:obj:`Optional[bool]`): Whether to update the total memory usage. Defaults to True.
        """
        path = layer_name.split(".")
        assert len(path) >= 2, f"Only support deleting non-root layers, layer_name: {layer_name}"

        parent_path = path[0:-1]
        layer = path[-1]
        parent = self.find_layer_state(parent_path)

        if parent is not None and layer in parent.sub_model_stats:
            del parent.sub_model_stats[layer]

        if flush:
            self.update_total_memory()

    def update_total_memory(self) -> None:
        """
        Overview:
            Update the total memory usage of the model and sub-models.
        """
        self._total_mem = self._layer_mem

        for stat in self.sub_model_stats.values():
            # Update sub-model status first.
            stat.update_total_memory()
            # Add sub-model total_mem to model total_mem.
            self._total_mem += stat._total_mem

    def find_layer_state(self, path: Tuple[str], create: bool = False) -> "SimpleMemState":
        """
        Overview:
            Find the memory state of a layer.
        Arguments:
            - path (:obj:`Tuple[str]`): The path to the layer.
            - create (:obj:`Optional[bool]`): Whether to create the layer if it doesn't exist. Defaults to False.
        Returns:
            - state (:obj:`SimpleMemState`): The memory state of the layer.
        """
        current_node = self

        for _node in path:
            if _node not in current_node.sub_model_stats:
                if not create:
                    return None
                # Create a layer node.
                current_node.sub_model_stats[_node] = SimpleMemState(_node)

            current_node = current_node.sub_model_stats[_node]

        return current_node

    def dump(self, prefix: str = "") -> str:
        """
        Overview:
            Dump the memory state of the model and sub-models.
        Arguments:
            - prefix (:obj:`Optional[str]`): The prefix to add to the layer names. Defaults to "".
        Returns:
            - result (:obj:`str`): The memory state information.
        """
        cur_prefix = prefix + "." + self.layer_name if prefix != "" else self.layer_name
        res = f"layer: {cur_prefix}, layer_mem: {self.layer_mem / MegaByte:.2f} MB, total_mem: {self.total_mem / MegaByte:.2f} MB\n"  # noqa

        for sub_layer in self.sub_model_stats.values():
            res += sub_layer.dump(cur_prefix)

        return res

    def to_json(self, base: int = 1024 * 1024) -> dict:
        """
        Overview:
            Convert the memory state to a JSON structure.
        Arguments:
            - base (:obj:`Optional[int]`): The base value to convert the memory usage to. Defaults to 1024 * 1024, \
                which converts the memory usage to MB.
        Returns:
            - result (:obj:`dict`): The JSON structure of the memory state.
        """
        children = [child.to_json() for child in self.sub_model_stats.values()]
        if len(children) == 0:
            return {"name": self.layer_name, "value": self.layer_mem // base}
        else:
            return {"name": self.layer_name, "children": children}


class ActivationMemState:
    """
    Overview:
        A class to represent the memory state of activation tensors.
    Properties:
        ``total_mem``
    Interfaces:
        ``add``, ``dump``, ``to_json``
    """

    def __init__(self, num_chunks: int) -> None:
        """
        Overview:
            Initialize the memory state of activation tensors.
        Arguments:
            - num_chunks (:obj:`int`): The number of chunks, multiple chunks are used in some large-scale models.
        """
        self._num_chunks = num_chunks

        self.inited: List[bool] = [False for _ in range(num_chunks)]
        self.states: List[SimpleMemState] = [SimpleMemState(f"activations_{idx}") for idx in range(num_chunks)]

    @property
    def total_mem(self) -> int:
        """
        Overview:
            Get the total memory usage of the activation tensors.
        Returns:
            - total_mem (:obj:`int`): The total memory usage of the activation tensors in bytes.
        """
        return sum(state.total_mem for state in self.states)

    def dump(self, prefix: str = "") -> str:
        """
        Overview:
            Dump the memory state of the activation tensors.
        Arguments:
            - prefix (:obj:`Optional[str]`): The prefix to add to the layer names. Defaults to "".
        Returns:
            - result (:obj:`str`): The memory state information.
        """
        return reduce(lambda x, y: x + y, [state.dump(prefix) for state in self.states])

    def to_json(self, base: int = 1024 * 1024) -> List[dict]:
        """
        Overview:
            Convert the memory state to a JSON structure.
        Arguments:
            - base (:obj:`Optional[int]`): The base value to convert the memory usage to. Defaults to 1024 * 1024, \
                which converts the memory usage to MB.
        Returns:
            - result (:obj:`List[dict]`): The JSON structure of the memory state.
        """
        return [state.to_json(base) for state in self.states]


def _unpack_naive_wrapper(model: torch.nn.Module) -> Tuple[torch.nn.Module, int]:
    num_chunks = len(model) if isinstance(model, torch.nn.ModuleList) else 1

    return model, num_chunks


class SimpleMemoryProfiler:
    """
    Overview:
        A memory profiler for a PyTorch neural network model.
    Interfaces:
        ``point``, ``step``
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        log_folder: str,
        total_steps: int = 5,
    ):
        """
        Overview:
            Initialize the memory profiler.
        Arguments:
            - model (:obj:`torch.nn.Module`): The model to profile.
            - optimizer (:obj:`torch.optim.Optimizer`): The optimizer used for training the model.
            - log_folder (:obj:`str`): The folder to write the memory state information to.
            - total_steps (:obj:`Optional[int]`): The number of steps to trace. Defaults to 5.
        """
        self._model, self._num_model_chunks = _unpack_naive_wrapper(model)
        self._optimizer = optimizer
        self._log_folder = log_folder
        self._remaining_steps = total_steps

        self._stoped = False
        self._record_start_time = time.time()

        # For activation memory state.

        self._activation_mem: int = 0
        self._activation_mem_max: int = 0
        self._activation_base_mems = ActivationMemState(self._num_model_chunks)

        # Check or create log folder
        os.makedirs(self._log_folder, exist_ok=True)

        # Register activation memory tracking hooks
        if self._num_model_chunks > 1:
            for chunk_id in range(self._num_model_chunks):
                self._register_activation_trace_hooks(chunk_id, self._model[chunk_id])
        else:
            self._register_activation_trace_hooks(0, self._model)

        # Calculate static parameter cuda memory
        self._param_mem_state = SimpleMemState("param_mem")
        self._calc_tensor_memory(self._param_mem_state, self._model.named_parameters())
        # Calculate static grad cuda memory
        self._grad_mem_state = SimpleMemState("grad_mem")
        self._calc_tensor_memory(self._grad_mem_state, self._model.named_parameters(), True)
        # Calculate static optimizer state cuda memory
        self._os_params_mem_state = SimpleMemState("os_params_mem")
        self._os_state_mem_state = SimpleMemState("os_state_mem")
        self._calc_tensor_group_memory(self._os_params_mem_state, list(enumerate(self._optimizer.param_groups)))

        # Generate the first memory record
        self.point(with_options="params,grads,os_params", create=True)

    def point(self, with_options: str = "", create: bool = False) -> None:
        """
        Overview:
            Record the memory state of the model and optimizer at current point.
        Arguments:
            - with_options (:obj:`Optional[str]`): The options to include in the memory state. Defaults to "".
            - create (:obj:`Optional[bool]`): Whether to create a new memory record. Defaults to False.
        """
        now = time.time()
        file = f"{self._log_folder}/memory.log"

        if with_options == "all":
            options = ["params", "grads", "os_params", "os_state", "activation_base"]
        else:
            options = with_options.split(",")

        total_mem = (
            self._param_mem_state.total_mem + self._grad_mem_state.total_mem + self._os_params_mem_state.total_mem +
            self._os_state_mem_state.total_mem + self._activation_mem
        ) / MegaByte

        # Generate summary information for memory state
        summary_info = (
            f"total_memory: {total_mem:.2f} MB" + "\n" +
            f"params_memory: {self._param_mem_state.total_mem / MegaByte:.2f} MB, " +
            f"grads_memory: {self._grad_mem_state.total_mem / MegaByte:.2f} MB, " +
            f"os_params_memory: {self._os_params_mem_state.total_mem / MegaByte:.2f} MB, " +
            f"os_state_memory: {self._os_state_mem_state.total_mem / MegaByte:.2f} MB, " +
            f"activation_memory: {self._activation_mem / MegaByte:.2f} MB"
        )

        # Generate layout information based on selected options
        layout_info = ""
        if "params" in options:
            layout_info += "params_layout:\n" + self._param_mem_state.dump()
        if "grads" in options:
            layout_info += "grads_layout:\n" + self._grad_mem_state.dump()
        if "os_params" in options:
            layout_info += "os_params_layout:\n" + self._os_params_mem_state.dump()
        if "os_state" in options:
            layout_info += "os_state_layout:\n" + self._os_state_mem_state.dump()
        if "activation_base" in options:
            layout_info += "activation_base_layout:\n" + self._activation_base_mems.dump()

        # Write memory state information to log file
        file_mode = "w" if create else "a"
        with open(file, file_mode, encoding="utf-8") as writer:
            writer.write(
                "Memory State:\n" + f"time: {now - self._record_start_time}\n" + "---summary---\n" + summary_info + "\n"
            )
            if layout_info != "":
                writer.write("---Layout---\n" + layout_info)
            writer.write("\n")

    def step(self) -> None:
        """
        Overview:
            Update the memory state of the optimizer state (e.g., momentum, learning rate) and record the memory state.
        """
        if self._stoped:
            return

        self._remaining_steps -= 1
        if self._remaining_steps == 0:
            self._stoped = True

        # Update os state memory usage
        self._os_state_mem_state = SimpleMemState("os_state_mem")
        self._calc_tensor_group_memory(self._os_state_mem_state, list(self._optimizer.state_dict()["state"].items()))

        if not self._stoped:
            # Do we need to print os_state_layout every time? Is it always constant?
            self.point(with_options="os_state")
        else:
            # Dump memory layout
            self.point(with_options="all")
            # Generate sunburst charts
            self._render_sunburst_chart(self._param_mem_state.to_json()["children"], "params_memory_sunburst")
            self._render_sunburst_chart(self._grad_mem_state.to_json()["children"], "grads_memory_sunburst")
            self._render_sunburst_chart(
                [self._os_params_mem_state.to_json(),
                 self._os_state_mem_state.to_json()],
                "os_memory_sunburst",
            )
            self._render_sunburst_chart(self._activation_base_mems.to_json(), "activation_memory_sunburst")
            # Generate summary sunburst chart
            summary_sunburst_data = [
                {
                    "name": "params",
                    "value": self._param_mem_state.total_mem // MegaByte
                },
                {
                    "name": "grads",
                    "value": self._grad_mem_state.total_mem // MegaByte
                },
                {
                    "name": "os_params",
                    "value": self._os_params_mem_state.total_mem // MegaByte
                },
                {
                    "name": "os_state",
                    "value": self._os_state_mem_state.total_mem // MegaByte
                },
                {
                    "name": "activation",
                    "value": self._activation_mem_max // MegaByte
                },
            ]

            self._render_sunburst_chart(summary_sunburst_data, "summary_sunburst")

    def _render_sunburst_chart(self, data: Any, name: str) -> None:
        """
        Overview:
            Render a sunburst chart for the memory state with pyecharts.
        Arguments:
            - data (:obj:`Any`): The data to render.
            - name (:obj:`str`): The name of the chart.
        """
        pyecharts.charts.Sunburst(init_opts=pyecharts.options.InitOpts(width="1000px", height="1000px")).add(
            name,
            data_pair=data,
            highlight_policy="ancestor",
            radius=[0, "95%"],
            levels=[
                {},
                {
                    "r0": "10%",
                    "r": "35%",
                    "itemStyle": {
                        "borderWidth": 3
                    },
                    "label": {
                        "align": "left"
                    },
                },
                {
                    "r0": "35%",
                    "r": "55%",
                    "label": {
                        "align": "left"
                    }
                },
                {
                    "r0": "55%",
                    "r": "70%",
                    "label": {
                        "align": "left"
                    }
                },
                {
                    "r0": "70%",
                    "r": "80%",
                    "label": {
                        "align": "left"
                    }
                },
                {
                    "r0": "80%",
                    "r": "90%",
                    "label": {
                        "align": "left"
                    }
                },
                {
                    "r0": "90%",
                    "r": "92%",
                    "label": {
                        "position": "outside",
                        "padding": 3,
                        "silent": False
                    },
                    "itemStyle": {
                        "borderWidth": 3
                    },
                },
            ],
        ).set_global_opts(title_opts=pyecharts.options.TitleOpts(title="CUDA Memory")
                          ).set_series_opts(label_opts=pyecharts.options.LabelOpts(formatter="{b}")
                                            ).render(f"{self._log_folder}/{name}.html")

    def _inner_activation_trace_hook(
            self,
            chunk_id: int,
            layer_name: str,
            model: Any,
            inputs: Any,
            output: torch.Tensor,
    ) -> None:
        """
        Overview:
            Hook function to trace the activation memory usage for a inner layer.

        .. note::
            For more details about hook mechanism, please refer to the PyTorch documentation.

        Arguments:
            - chunk_id (:obj:`int`): The model chunk id.
            - layer_name (:obj:`str`): The name of the layer.
            - model (:obj:`Any`): The model to trace.
            - inputs (:obj:`Any`): The inputs to the layer.
            - output (:obj:`torch.Tensor`): The output tensor.
        """
        del model, inputs
        assert isinstance(output, torch.Tensor), f"Invalid output type: {type(output)}"

        if self._stoped or self._activation_base_mems.inited[chunk_id]:
            return

        # Delay updating the total_mem of activation_base_mem here, it will be handled in the forward ending hook.
        self._activation_base_mems.states[chunk_id].add(
            layer_name, output.element_size() * output.nelement(), flush=False
        )

    def _activation_trace_hook_forward(self, chunk_id: int, model: Any, inputs: Any, output: Any) -> None:
        """
        Overview:
            Hook function to trace the activation memory usage for a forward pass.

        .. note::
            For more details about hook mechanism, please refer to the PyTorch documentation.

        Arguments:
            - chunk_id (:obj:`int`): The model chunk id.
            - model (:obj:`Any`): The model to trace.
            - inputs (:obj:`Any`): The inputs to the model.
            - output (:obj:`Any`): The output of the model.
        """
        del model, inputs

        if self._stoped:
            return

        # Check if the activation memory has been initialized
        if self._activation_base_mems.inited[chunk_id] is False:
            self._activation_base_mems.inited[chunk_id] = True
            # Update the total memory of the activation base memory state
            self._activation_base_mems.states[chunk_id].update_total_memory()
            # Set with_options to "activation_base" to include activation_base_layout in the memory dump
            with_options = "activation_base"
        else:
            with_options = ""

        # Accumulate activation memory usage for each forward pass
        self._activation_mem += self._activation_base_mems.states[chunk_id].total_mem
        if self._activation_mem > self._activation_mem_max:
            self._activation_mem_max = self._activation_mem

        # Trigger a memory record
        self.point(with_options)

    def _activation_tarce_hook_backward(self, chunk_id: int, model: Any, inputs: Any, grad_outputs: Any) -> None:
        """
        Overview:
            Hook function to trace the activation memory usage for a backward pass.

        .. note::
            For more details about hook mechanism, please refer to the PyTorch documentation.

        Arguments:
            - chunk_id (:obj:`int`): The model chunk id.
            - model (:obj:`Any`): The model to trace.
            - inputs (:obj:`Any`): The inputs to the model.
            - grad_outputs (:obj:`Any`): The gradients of the outputs.
        """
        del model, inputs, grad_outputs

        if self._stoped:
            return

        # Release activation memory usage for each backward pass
        self._activation_mem -= self._activation_base_mems.states[chunk_id].total_mem

        # Trigger a memory record
        self.point()

    def _register_activation_trace_hooks(self, chunk_id: int, model_chunk: torch.nn.Module) -> None:
        """
        Overview:
            Register activation trace hooks for the model and each submodule in the model.
        Arguments:
            - chunk_id (:obj:`int`): The model chunk id.
            - model_chunk (:obj:`torch.nn.Module`): The model chunk to trace.
        """

        # Register inner activation trace hooks for each submodule in the model
        for layer_name, sub_model in model_chunk.named_modules():
            # Register the hook
            if len(sub_model._modules) != 0:
                continue  # TODO: in some special cases, we may need some additional configuration to correct

            sub_model.register_forward_hook(partial(self._inner_activation_trace_hook, chunk_id, layer_name))

        # Register a forward hook for the main model to track activation memory usage
        model_chunk.register_forward_hook(partial(self._activation_trace_hook_forward, chunk_id))
        # Register a backward hook for the main model to release activation memory usage
        model_chunk.register_full_backward_hook(partial(self._activation_tarce_hook_backward, chunk_id))

    def _calc_tensor_memory(
            self,
            root_stat: SimpleMemState,
            named_tensors: Dict[str, torch.Tensor],
            require_grad: bool = False
    ) -> None:
        """
        Overview:
            Core function to calculate the memory usage of tensors and update the memory state.
        Arguments:
            - root_stat (:obj:`SimpleMemState`): The root memory state.
            - named_tensors (:obj:`Dict[str, torch.Tensor]`): A dictionary containing the named tensors.
            - require_grad (:obj:`Optional[bool]`): Whether to consider tensors with gradients. Defaults to False.
        """
        for name, tensor in named_tensors:
            if require_grad and not tensor.requires_grad:
                continue

            layer_splits = name.split(sep=".")
            layer_stat = root_stat.find_layer_state(layer_splits, create=True)
            layer_stat.layer_mem = tensor.element_size() * tensor.nelement()

        root_stat.update_total_memory()

    def _calc_tensor_group_memory(self, root_stat: SimpleMemState, tensor_groups: List[Tuple[int, torch.Tensor]]):
        """
        Overview:
            Core function to calculate the memory usage of a group of tensors and update the memory state.
        Arguments:
            - root_stat (:obj:`SimpleMemState`): The root memory state.
            - tensor_groups (:obj:`List[Tuple[int, torch.Tensor]]`): A list of tuples containing the tensor groups.
        """

        def _normalize_helper(named_tensors: Dict[str, Any]) -> List[Tuple[str, Any]]:
            res = {}

            for name, tensors in named_tensors.items():
                if isinstance(tensors, torch.Tensor):
                    res[name] = tensors
                elif isinstance(tensors, (list, tuple)):
                    for index, tensor in enumerate(tensors):
                        res[f"{name}.{index}"] = tensor
                elif isinstance(tensors, dict):
                    for subname, tensor in tensors.items():
                        res[f"{name}.{subname}"] = tensor
                else:
                    raise TypeError(f"unsupported normalize value type: {type(tensors)}")

            return list(res.items())

        def _value_check(tensor_or_tensors):
            if torch.is_tensor(tensor_or_tensors):
                return True
            elif isinstance(tensor_or_tensors, (list, tuple)) and all(torch.is_tensor(x) for x in tensor_or_tensors):
                return True
            elif isinstance(tensor_or_tensors, dict) and all(torch.is_tensor(x) for x in tensor_or_tensors.values()):
                return True
            else:
                return False

        # Calculate the memory usage of a group of tensors.
        for idx, tensors in tensor_groups:
            # Normalize the named tensors
            named_tensors = {f"{idx}.{k}": v for k, v in tensors.items() if _value_check(v)}
            named_tensors = _normalize_helper(named_tensors)
            # Calculate the memory usage of the tensors and update the memory state
            self._calc_tensor_memory(root_stat, named_tensors)


def get_current_device() -> torch.device:
    """
    Overview:
        Get the current PyTorch tensor device.

    Returns:
        - device (:obj:`torch.device`): The current device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def multi_chunk_test():
    """
    Overview:
        A test function to demonstrate the memory profiler for a model with multiple chunks.
    """

    class SimpleModel(torch.nn.Module):

        def __init__(self, skip_layer2: bool = False):
            super().__init__()
            self.layer1 = torch.nn.Linear(5120, 5120, True)
            self.layer3 = torch.nn.Linear(5120, 5120, False)

            if skip_layer2:
                self.layer2 = None
            else:
                self.layer2 = SimpleModel(skip_layer2=True)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            output1 = self.layer1(inputs)
            if self.layer2 is not None:
                output2 = self.layer2(output1)
            else:
                output2 = output1
            output = self.layer3(output2)

            return output

    def _simple_schedule(_num_chunks, _model_chunks, _input) -> torch.Tensor:
        if _num_chunks > 1:
            _output = _input
            for _model_chunk in _model_chunks:
                _output = _model_chunk(_output)
        else:
            _output = _model_chunks(_input)

        return _output

    # num_chunks config
    _num_chunks = 1

    # init model and optimizer
    if _num_chunks > 1:
        _chunks = [SimpleModel(skip_layer2=idx % 2 == 0) for idx in range(_num_chunks)]
        _model = torch.nn.ModuleList(_chunks).to(get_current_device())
    else:
        _model: torch.nn.Module = SimpleModel().to(get_current_device())
    _optimizer = torch.optim.Adam(_model.parameters())

    # init profiler
    profiler = SimpleMemoryProfiler(_model, _optimizer, "./test_simple_memory_profiler_multi_chunk", total_steps=1)

    _optimizer.zero_grad()

    # inputs
    x1 = torch.randn((128, 5120)).to(get_current_device())
    x2 = torch.randn((128, 5120)).to(get_current_device())
    # forward
    out1 = _simple_schedule(_num_chunks, _model, x1)
    out2 = _simple_schedule(_num_chunks, _model, x2)
    # backward
    out1.mean().backward()
    out2.mean().backward()

    _optimizer.step()

    # Update the optimizer state memory usage and record the memory state
    profiler.step()


if __name__ == "__main__":
    multi_chunk_test()
