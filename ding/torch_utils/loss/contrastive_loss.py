from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.utils import SequenceType


class ContrastiveLoss(nn.Module):
    """
    Overview:
        The class for contrastive learning losses. Only InfoNCE loss is supported currently. \
        Code Reference: https://github.com/rdevon/DIM. Paper Reference: https://arxiv.org/abs/1808.06670.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(
            self,
            x_size: Union[int, SequenceType],
            y_size: Union[int, SequenceType],
            heads: SequenceType = [1, 1],
            encode_shape: int = 64,
            loss_type: str = "infoNCE",  # Only the InfoNCE loss is available now.
            temperature: float = 1.0,
    ) -> None:
        """
        Overview:
            Initialize the ContrastiveLoss object using the given arguments.
        Arguments:
            - x_size (:obj:`Union[int, SequenceType]`): input shape for x, both the obs shape and the encoding shape \
                are supported.
            - y_size (:obj:`Union[int, SequenceType]`): Input shape for y, both the obs shape and the encoding shape \
                are supported.
            - heads (:obj:`SequenceType`): A list of 2 int elems, ``heads[0]`` for x and ``head[1]`` for y. \
                Used in multi-head, global-local, local-local MI maximization process.
            - encoder_shape (:obj:`Union[int, SequenceType]`): The dimension of encoder hidden state.
            - loss_type: Only the InfoNCE loss is available now.
            - temperature: The parameter to adjust the ``log_softmax``.
        """
        super(ContrastiveLoss, self).__init__()
        assert len(heads) == 2, "Expected length of 2, but got: {}".format(len(heads))
        assert loss_type.lower() in ["infonce"]

        self._type = loss_type.lower()
        self._encode_shape = encode_shape
        self._heads = heads
        self._x_encoder = self._create_encoder(x_size, heads[0])
        self._y_encoder = self._create_encoder(y_size, heads[1])
        self._temperature = temperature

    def _create_encoder(self, obs_size: Union[int, SequenceType], heads: int) -> nn.Module:
        """
        Overview:
            Create the encoder for the input obs.
        Arguments:
            - obs_size (:obj:`Union[int, SequenceType]`): input shape for x, both the obs shape and the encoding shape \
                are supported. If the obs_size is an int, it means the obs is a 1D vector. If the obs_size is a list \
                such as [1, 16, 16], it means the obs is a 3D image with shape [1, 16, 16].
            - heads (:obj:`int`): The number of heads.
        Returns:
            - encoder (:obj:`nn.Module`): The encoder module.
        Examples:
            >>> obs_size = 16
            or
            >>> obs_size = [1, 16, 16]
            >>> heads = 1
            >>> encoder = self._create_encoder(obs_size, heads)
        """
        from ding.model import ConvEncoder, FCEncoder

        if isinstance(obs_size, int):
            obs_size = [obs_size]
        assert len(obs_size) in [1, 3]

        if len(obs_size) == 1:
            hidden_size_list = [128, 128, self._encode_shape * heads]
            encoder = FCEncoder(obs_size[0], hidden_size_list)
        else:
            hidden_size_list = [32, 64, 64, self._encode_shape * heads]
            if obs_size[-1] >= 36:
                encoder = ConvEncoder(obs_size, hidden_size_list)
            else:
                encoder = ConvEncoder(obs_size, hidden_size_list, kernel_size=[4, 3, 2], stride=[2, 1, 1])
        return encoder

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Computes the noise contrastive estimation-based loss, a.k.a. infoNCE.
        Arguments:
            - x (:obj:`torch.Tensor`): The input x, both raw obs and encoding are supported.
            - y (:obj:`torch.Tensor`): The input y, both raw obs and encoding are supported.
        Returns:
            loss (:obj:`torch.Tensor`): The calculated loss value.
        Examples:
            >>> x_dim = [3, 16]
            >>> encode_shape = 16
            >>> x = np.random.normal(0, 1, size=x_dim)
            >>> y = x ** 2 + 0.01 * np.random.normal(0, 1, size=x_dim)
            >>> estimator = ContrastiveLoss(dims, dims, encode_shape=encode_shape)
            >>> loss = estimator.forward(x, y)
        Examples:
            >>> x_dim = [3, 1, 16, 16]
            >>> encode_shape = 16
            >>> x = np.random.normal(0, 1, size=x_dim)
            >>> y = x ** 2 + 0.01 * np.random.normal(0, 1, size=x_dim)
            >>> estimator = ContrastiveLoss(dims, dims, encode_shape=encode_shape)
            >>> loss = estimator.forward(x, y)
        """

        N = x.size(0)
        x_heads, y_heads = self._heads
        x = self._x_encoder.forward(x).view(N, x_heads, self._encode_shape)
        y = self._y_encoder.forward(y).view(N, y_heads, self._encode_shape)

        x_n = x.view(-1, self._encode_shape)
        y_n = y.view(-1, self._encode_shape)

        # Use inner product to obtain positive samples.
        # [N, x_heads, encode_dim] * [N, encode_dim, y_heads] -> [N, x_heads, y_heads]
        u_pos = torch.matmul(x, y.permute(0, 2, 1)).unsqueeze(2)
        # Use outer product to obtain all sample permutations.
        # [N * x_heads, encode_dim] X [encode_dim, N * y_heads] -> [N * x_heads, N * y_heads]
        u_all = torch.mm(y_n, x_n.t()).view(N, y_heads, N, x_heads).permute(0, 2, 3, 1)

        # Mask the diagonal part to obtain the negative samples, with all diagonals setting to -10.
        mask = torch.eye(N)[:, :, None, None].to(x.device)
        n_mask = 1 - mask
        u_neg = (n_mask * u_all) - (10. * (1 - n_mask))
        u_neg = u_neg.view(N, N * x_heads, y_heads).unsqueeze(dim=1).expand(-1, x_heads, -1, -1)

        # Concatenate positive and negative samples and apply log softmax.
        pred_lgt = torch.cat([u_pos, u_neg], dim=2)
        pred_log = F.log_softmax(pred_lgt * self._temperature, dim=2)

        # The positive score is the first element of the log softmax.
        loss = -pred_log[:, :, 0, :].mean()
        return loss
