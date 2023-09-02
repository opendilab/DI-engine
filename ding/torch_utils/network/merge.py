"""
This file provides an implementation of several different neural network modules that are used for merging and
transforming input data in various ways. The following components can be used when we are dealing with
data from multiple modes, or when we need to merge multiple intermediate embedded representations in
the forward process of a model.

The main classes defined in this code are:

    - BilinearGeneral: This class implements a bilinear transformation layer that applies a bilinear transformation to
        incoming data, as described in the "Multiplicative Interactions and Where to Find Them", published at ICLR 2020,
        https://openreview.net/forum?id=rylnK6VtDH. The transformation involves two input features and an output
        feature, and also includes an optional bias term.

    - TorchBilinearCustomized: This class implements a bilinear layer similar to the one provided by PyTorch
        (torch.nn.Bilinear), but with additional customizations. This class can be used as an alternative to the
        BilinearGeneral class.

    - TorchBilinear: This class is a simple wrapper around the PyTorch's built-in nn.Bilinear module. It provides the
        same functionality as PyTorch's nn.Bilinear but within the structure of the current module.

    - FiLM: This class implements a Feature-wise Linear Modulation (FiLM) layer. FiLM layers apply an affine
        transformation to the input data, conditioned on some additional context information.

    - GatingType: This is an enumeration class that defines different types of gating mechanisms that can be used in
        the modules.

    - SumMerge: This class provides a simple summing mechanism to merge input streams.

    - VectorMerge: This class implements a more complex merging mechanism for vector streams.
        The streams are first transformed using layer normalization, a ReLU activation, and a linear layer.
        Then they are merged either by simple summing or by using a gating mechanism.

The implementation of these classes involves PyTorch and Numpy libraries, and the classes use PyTorch's nn.Module as
the base class, making them compatible with PyTorch's neural network modules and functionalities.
These modules can be useful building blocks in more complex deep learning architectures.
"""

import enum
import math
from collections import OrderedDict
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BilinearGeneral(nn.Module):
    """
    Overview:
        Bilinear implementation as in:
        Multiplicative Interactions and Where to Find Them, ICLR 2020, https://openreview.net/forum?id=rylnK6VtDH
    Arguments:
        - in1_features (:obj:`int`): size of each first input sample
        - in2_features (:obj:`int`): size of each second input sample
        - out_features (:obj:`int`): size of each output sample
    """

    def __init__(self, in1_features, in2_features, out_features):
        super(BilinearGeneral, self).__init__()
        # Initialize the weight matrices W and U, and the bias vectors V and b
        self.W = nn.Parameter(torch.Tensor(out_features, in1_features, in2_features))
        self.U = nn.Parameter(torch.Tensor(out_features, in2_features))
        self.V = nn.Parameter(torch.Tensor(out_features, in1_features))
        self.b = nn.Parameter(torch.Tensor(out_features))
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.in1_features)
        self.W.data.uniform_(-stdv, stdv)
        self.U.data.uniform_(-stdv, stdv)
        self.V.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, x, z):
        # Compute the bilinear function
        # x^TWz
        out_W = torch.einsum('bi,kij,bj->bk', x, self.W, z)
        # x^TU
        out_U = z.matmul(self.U.t())
        # Vz
        out_V = x.matmul(self.V.t())
        # x^TWz + x^TU + Vz + b
        out = out_W + out_U + out_V + self.b
        return out


class TorchBilinearCustomized(nn.Module):
    """
    Overview:
        Customized Torch Bilinear implementation.
    Arguments:
        - in1_features (:obj:`int`): size of each first input sample
        - in2_features (:obj:`int`): size of each second input sample
        - out_features (:obj:`int`): size of each output sample
    """

    def __init__(self, in1_features, in2_features, out_features):
        super(TorchBilinearCustomized, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in1_features, in2_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in1_features)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, z):
        # Using torch.einsum for the bilinear operation
        out = torch.einsum('bi,oij,bj->bo', x, self.weight, z) + self.bias
        return out.squeeze(-1)


"""
Overview:
    Implementation of the Bilinear layer as in PyTorch:
    https://pytorch.org/docs/stable/generated/torch.nn.Bilinear.html#torch.nn.Bilinear
Arguments:
    - in1_features (:obj:`int`): size of each first input sample
    - in2_features (:obj:`int`): size of each second input sample
    - out_features (:obj:`int`): size of each output sample
    - bias (:obj:`bool`): If set to False, the layer will not learn an additive bias. Default: ``True``.
"""
TorchBilinear = nn.Bilinear


class FiLM(nn.Module):
    """
    Overview:
        Feature-wise Linear Modulation (FiLM) Layer.
        This layer applies feature-wise affine transformation based on context.
    Arguments:
        - feature_dim (:obj:`int`). The dimension of the input feature vector.
        - context_dim (:obj:`int`). The dimension of the input context vector.
    """

    def __init__(self, feature_dim, context_dim):
        super(FiLM, self).__init__()
        # Define the fully connected layer for context
        # The output dimension is twice the feature dimension for gamma and beta
        self.context_layer = nn.Linear(context_dim, 2 * feature_dim)

    def forward(self, feature, context):
        """
        Overview:
            Forward propagation.
        Arguments:
            - feature (:obj:`torch.Tensor`). The input feature, shape (batch_size, feature_dim)
            - context (:obj:`torch.Tensor`). The input context, shape (batch_size, context_dim)
        Returns:
            - conditioned_feature : torch.Tensor. The output feature after FiLM, shape (batch_size, feature_dim)
        """
        # Pass context through the fully connected layer
        out = self.context_layer(context)
        # Split the output into two parts: gamma and beta
        # The dimension for splitting is 1 (feature dimension)
        gamma, beta = torch.split(out, out.shape[1] // 2, dim=1)
        # Apply feature-wise affine transformation
        conditioned_feature = gamma * feature + beta
        return conditioned_feature


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
