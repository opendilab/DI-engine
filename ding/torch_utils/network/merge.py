"""
This file provides two components for consolidating data streams, SumMerge and VectorMerge.

The following components can be used when we are dealing with data from multiple modes,
or when we need to merge multiple intermediate embedded representations in the forward process of a model.

While SumMerge simply sums multiple data streams in the first dimension,
VectorMerge provides three more complex weighted summations.
"""

import enum
from typing import List, Dict
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GatingType(enum.Enum):
    r"""
    Overview:
        Defines how the tensors are gated and aggregated in modules.
    """
    NONE = 'none'
    GLOBAL = 'global'
    POINTWISE = 'pointwise'


class SumMerge(nn.Module):
    r"""
    Overview:
        Merge streams using a simple sum.
        Streams must have the same size.
        This module can merge any type of stream (vector, units or visual).
    """

    def forward(self, tensors: List[Tensor]):
        # stack the tensors along the first dimension
        stacked = torch.stack(tensors, dim=0)

        # compute the sum along the first dimension
        summed = torch.sum(stacked, dim=0)
        # summed = sum(tensors)
        return summed


class VectorMerge(nn.Module):
    r"""
    Overview:
        Merge vector streams.

        Streams are first transformed through layer normalization, relu and linear
        layers, then summed, so they don't need to have the same size.
        Gating can also be used before the sum.

        If gating_type is not none, the sum is weighted using a softmax
        of the intermediate activations labelled above.

        Sepcifically,
            GatingType.NONE:
                Means simple addition of streams and the sum is not weighted based on gate features.
            GatingType.GLOBAL:
                Each data stream is weighted by a global gate value, and the sum of all global gate values is 1.
            GatingType.POINTWISE:
                Compared to GLOBAL, each value in the data stream feature tensor is weighted.
    """

    def __init__(
        self,
        input_sizes: Dict[str, int],
        output_size: int,
        gating_type: GatingType = GatingType.NONE,
        use_layer_norm: bool = True,
    ):
        r"""
        Overview:
            Initializes VectorMerge module.
        Arguments:
            - input_sizes: A dictionary mapping input names to their size (a single
                integer for 1d inputs, or None for 0d inputs).
                If an input size is None, we assume it's ().
            - output_size: The size of the output vector.
            - gating_type: The type of gating mechanism to use.
            - use_layer_norm: Whether to use layer normalization.
        """
        super().__init__()
        self._input_sizes = OrderedDict(input_sizes)
        self._output_size = output_size
        self._gating_type = gating_type
        self._use_layer_norm = use_layer_norm

        if self._use_layer_norm:
            self._layer_norms = nn.ModuleDict()
        else:
            self._layer_norms = None

        self._linears = nn.ModuleDict()
        for name, size in self._input_sizes.items():
            linear_input_size = size if size > 0 else 1
            if self._use_layer_norm:
                self._layer_norms[name] = nn.LayerNorm(linear_input_size)
            self._linears[name] = nn.Linear(linear_input_size, self._output_size)

        self._gating_linears = nn.ModuleDict()
        if self._gating_type is GatingType.GLOBAL:
            self.gate_size = 1
        elif self._gating_type is GatingType.POINTWISE:
            self.gate_size = self._output_size
        elif self._gating_type is GatingType.NONE:
            self._gating_linears = None
        else:
            raise ValueError(f'Gating type {self._gating_type} is not supported')

        if self._gating_linears is not None:
            if len(self._input_sizes) == 2:
                # more efficient than the general version below
                for name, size in self._input_sizes.items():
                    gate_input_size = size if size > 0 else 1
                    gating_layer = nn.Linear(gate_input_size, self.gate_size)
                    torch.nn.init.normal_(gating_layer.weight, std=0.005)
                    torch.nn.init.constant_(gating_layer.bias, 0.0)
                    self._gating_linears[name] = gating_layer
            else:
                for name, size in self._input_sizes.items():
                    gate_input_size = size if size > 0 else 1
                    gating_layer = nn.Linear(gate_input_size, len(self._input_sizes) * self.gate_size)
                    torch.nn.init.normal_(gating_layer.weight, std=0.005)
                    torch.nn.init.constant_(gating_layer.bias, 0.0)
                    self._gating_linears[name] = gating_layer

    def encode(self, inputs: Dict[str, Tensor]):
        gates, outputs = [], []
        for name, size in self._input_sizes.items():
            feature = inputs[name]
            if size <= 0 and feature.dim() == 1:
                feature = feature.unsqueeze(-1)
            feature = feature.to(torch.float32)
            if self._use_layer_norm and name in self._layer_norms:
                feature = self._layer_norms[name](feature)
            feature = F.relu(feature)
            gates.append(feature)
            outputs.append(self._linears[name](feature))
        return gates, outputs

    def _compute_gate(
        self,
        init_gate: List[Tensor],
    ):
        if len(self._input_sizes) == 2:
            gate = [self._gating_linears[name](y) for name, y in zip(self._input_sizes.keys(), init_gate)]
            gate = sum(gate)
            sigmoid = torch.sigmoid(gate)
            gate = [sigmoid, 1.0 - sigmoid]
        else:
            gate = [self._gating_linears[name](y) for name, y in zip(self._input_sizes.keys(), init_gate)]
            gate = sum(gate)
            gate = gate.reshape([-1, len(self._input_sizes), self.gate_size])
            gate = F.softmax(gate, dim=1)
            assert gate.shape[1] == len(self._input_sizes)
            gate = [gate[:, i] for i in range(len(self._input_sizes))]
        return gate

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        gates, outputs = self.encode(inputs)
        if len(outputs) == 1:
            # Special case of 1-D inputs that do not need any gating.
            output = outputs[0]
        elif self._gating_type is GatingType.NONE:
            output = sum(outputs)
        else:
            gate = self._compute_gate(gates)
            data = [g * d for g, d in zip(gate, outputs)]
            output = sum(data)
        return output
