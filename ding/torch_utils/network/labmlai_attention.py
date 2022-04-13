# source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/e75e53bb03bc3ab68ce61699c0fcf280d4cfb3d6/labml_nn/transformers/xl/__init__.py#L47

"""
---
title: Transformer XL
summary: >
  Documented implementation with explanations of a
  Transformer-XL model.
---
# Transformer XL
This is an implementation of
[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://papers.labml.ai/paper/1901.02860)
in [PyTorch](https://pytorch.org).
Transformer has a limited attention span,
equal to the length of the sequence trained in parallel.
All these positions have a fixed positional encoding.
Transformer XL increases this attention span by letting
each of the positions pay attention to precalculated past embeddings.
For instance if the context length is $l$, it will keep the embeddings of
all layers for previous batch of length $l$ and feed them to current step.
If we use fixed-positional encodings these pre-calculated embeddings will have
the same positions as the current context.
They introduce relative positional encoding, where the positional encodings
are introduced at the attention calculation.
Annotated implementation of relative multi-headed attention is in [`relative_mha.py`](relative_mha.html).
Here's [the training code](experiment.html) and a notebook for training a transformer XL model on Tiny Shakespeare dataset.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/xl/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/d3b6760c692e11ebb6a70242ac1c0002)
"""

from typing import List, Optional

import torch
import torch.nn as nn

from labml_helpers.module import Module
from labml_nn.utils import clone_module_list
from labml_nn.transformers.mha import MultiHeadAttention


def shift_right(x: torch.Tensor):
    """
    This method shifts $i^{th}$ row of a matrix by $i$ columns.
    If the input is `[[1, 2 ,3], [4, 5 ,6], [7, 8, 9]]`, the shifted
    result would be `[[1, 2 ,3], [0, 4, 5], [9, 0, 7]]`.
    *Ideally we should mask out the lower triangle but it's ok for our purpose*.
    """

    # Concatenate a column of zeros
    zero_pad = x.new_zeros(x.shape[0], 1, *x.shape[2:])
    x_padded = torch.cat([x, zero_pad], dim=1)

    # Reshape and remove excess elements from the end
    x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])
    x = x_padded[:-1].view_as(x)

    #
    return x


class RelativeMultiHeadAttention(MultiHeadAttention):
    """
    ## Relative Multi-Head Attention Module
    We override [Multi-Head Attention](mha.html) module so we only need to
    write the `get_scores` method.
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        # The linear transformations do not need a bias since we
        # explicitly include it when calculating scores.
        # However having a bias for `value` might make sense.
        super().__init__(heads, d_model, dropout_prob, bias=False)

        # Number of relative positions
        self.P = 2 ** 12

        # Relative positional embeddings for key relative to the query.
        # We need $2P$ embeddings because the keys can be before or after the query.
        self.key_pos_embeddings = nn.Parameter(torch.zeros((self.P * 2, heads, self.d_k)), requires_grad=True)
        # Relative positional embedding bias for key relative to the query.
        self.key_pos_bias = nn.Parameter(torch.zeros((self.P * 2, heads)), requires_grad=True)
        # Positional embeddings for the query is independent of the position of the query
        self.query_pos_bias = nn.Parameter(torch.zeros((heads, self.d_k)), requires_grad=True)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        r"""
        ### Get relative attention scores
        With absolute attention
        \begin{align}
        A^{abs}_{j} &= lin_q(X^q_i + P_i)^\top lin_k(X^k_j + P_j) \\
                      &= \underset{\textcolor{lightgreen}{A}}{Q_i^\top K_j} +
                         \underset{\textcolor{lightgreen}{B}}{Q_i^\top U^K_j} +
                         \underset{\textcolor{lightgreen}{C}}{{U^Q_i}^\top K_j} +
                         \underset{\textcolor{lightgreen}{D}}{{U^Q_i}^\top U^K_j}
        \end{align}
        where $Q_i, K_j$, are linear transformations of
         original embeddings $X^q_i, X^k_j$
         and $U^Q_i, U^K_j$ are linear transformations of
         absolute positional encodings $P_i, P_j$.
        They reason out that the attention to a given key should be the same regardless of
        the position of query.
        Hence replace $\underset{\textcolor{lightgreen}{C}}{{U^Q_i}^\top K_j}$
        with a constant $\underset{\textcolor{lightgreen}{C}}{\textcolor{orange}{v^\top} K_j}$.
        For the second and third terms relative positional encodings are introduced.
        So $\underset{\textcolor{lightgreen}{B}}{Q_i^\top U^K_j}$ is
        replaced with $\underset{\textcolor{lightgreen}{B}}{Q_i^\top \textcolor{orange}{R_{i - j}}}$
        and $\underset{\textcolor{lightgreen}{D}}{{U^Q_i}^\top U^K_j}$
        with $\underset{\textcolor{lightgreen}{D}}{\textcolor{orange}{S_{i-j}}}$.
        \begin{align}
        A^{rel}_{i,j} &= \underset{\mathbf{\textcolor{lightgreen}{A}}}{Q_i^\top K_j} +
                         \underset{\mathbf{\textcolor{lightgreen}{B}}}{Q_i^\top \textcolor{orange}{R_{i - j}}} +
                         \underset{\mathbf{\textcolor{lightgreen}{C}}}{\textcolor{orange}{v^\top} K_j} +
                         \underset{\mathbf{\textcolor{lightgreen}{D}}}{\textcolor{orange}{S_{i-j}}}
        \end{align}
        """

        # $\textcolor{orange}{R_k}$
        key_pos_emb = self.key_pos_embeddings[self.P - key.shape[0]:self.P + query.shape[0]]
        # $\textcolor{orange}{S_k}$
        key_pos_bias = self.key_pos_bias[self.P - key.shape[0]:self.P + query.shape[0]]
        # $\textcolor{orange}{v^\top}$
        query_pos_bias = self.query_pos_bias[None, None, :, :]

        # ${(\textcolor{lightgreen}{\mathbf{A + C}})}_{i,j} =
        # Q_i^\top K_j +
        # \textcolor{orange}{v^\top} K_jZ$
        ac = torch.einsum('ibhd,jbhd->ijbh', query + query_pos_bias, key)
        # $\textcolor{lightgreen}{\mathbf{B'}_{i,k}} = Q_i^\top \textcolor{orange}{R_k}$
        b = torch.einsum('ibhd,jhd->ijbh', query, key_pos_emb)
        # $\textcolor{lightgreen}{\mathbf{D'}_{i,k}} = \textcolor{orange}{S_k}$
        d = key_pos_bias[None, :, None, :]
        # Shift the rows of $\textcolor{lightgreen}{\mathbf{(B' + D')}_{i,k}}$
        # to get $$\textcolor{lightgreen}{\mathbf{(B + D)}_{i,j} = \mathbf{(B' + D')}_{i,i - j}}$$
        bd = shift_right(b + d)
        # Remove extra positions
        bd = bd[:, -key.shape[0]:]

        # Return the sum $$
        # \underset{\mathbf{\textcolor{lightgreen}{A}}}{Q_i^\top K_j} +
        # \underset{\mathbf{\textcolor{lightgreen}{B}}}{Q_i^\top \textcolor{orange}{R_{i - j}}} +
        # \underset{\mathbf{\textcolor{lightgreen}{C}}}{\textcolor{orange}{v^\top} K_j} +
        # \underset{\mathbf{\textcolor{lightgreen}{D}}}{\textcolor{orange}{S_{i-j}}}
        # $$
        return ac + bd


class TransformerXLLayer(Module):
    """
    ## Transformer XL Layer
    The transformer XL model comprises of a number of these layers.
    """

    def __init__(self,
                 d_model: int,
                 self_attn: RelativeMultiHeadAttention,
                 dropout_prob: float):
        """
        * `d_model` is the token embedding size
        * `self_attn` is the [self attention module](relative_mha.html)
        * `feed_forward` is the feed forward module
        * `dropout_prob` is the probability of dropping out after self attention and FFN
        """
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def forward(self,
                x: torch.Tensor,
                mem: Optional[torch.Tensor],
                mask: torch.Tensor
                ):
        """
        * `x` is a tensor of the token level feature vectors of shape `[seq_len, batch_size, d_model]`
        * `mem` is a tensor of the past token level feature vectors of shape `[mem_len, batch_size, d_model]`
        * `mask` is a matrix of shape `[seq_len, mem_len + seq_len, batch_size]` or `[seq_len, mem_len + seq_len, 1]`.
        `mask[i, j]` is  true if token at `i` can see token at `j`.
        """
        # Normalize the vectors before doing self attention
        z = self.norm_self_attn(x)
        z = x
        # If there is memory
        if mem is not None:
            # Normalize it
            mem = self.norm_self_attn(mem)
            # Concatenate with `z`
            m_z = torch.cat((mem, z), dim=0)
        # Ignore if there is no memory
        else:
            m_z = z
        # Attention
        self_attn = self.self_attn(query=z, key=m_z, value=m_z, mask=mask)
        # Add the attention results
        x = x + self.dropout(self_attn)

        # Normalize for feed-forward
        z = self.norm_ff(x)
        # Pass through the feed-forward network
        ff = self.feed_forward(z)
        # Add the feed-forward results back
        x = x + self.dropout(ff)

        return x


class TransformerXL(Module):
    """
    ## Transformer XL Model
    This consists of multiple transformer XL layers
    """

    def __init__(self, layer: TransformerXLLayer, n_layers: int):
        super().__init__()
        # Make copies of the transformer layer
        self.layers = clone_module_list(layer, n_layers)
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, mem: List[torch.Tensor], mask: torch.Tensor):
        """
        * `x` is a tensor of the token embeddings vectors of shape `[seq_len, batch_size, d_model]`
        * `mem` is a list of tensors of the past token level feature vectors of shape
        `[mem_len, batch_size, d_model]`  for each layer
        * `mask` is the masking matrix
        """
        # List to store token level feature vectors,
        # which will become the memories for the next sequential batch.
        new_mem = []
        # Run through each transformer layer
        for i, layer in enumerate(self.layers):
            # Add to the list of feature vectors
            new_mem.append(x.detach())
            # Memory
            m = mem[i] if mem else None
            # Run through the transformer XL layer
            x = layer(x=x, mem=m, mask=mask)
        # Finally, normalize the vectors
        return self.norm(x), new_mem
