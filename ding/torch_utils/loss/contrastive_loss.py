from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.utils import SequenceType


class ContrastiveLoss(nn.Module):
    """
        The class for contrastive learning losses.
        Only InfoNCE loss supported currently.
        Code Reference: https://github.com/rdevon/DIM.
        paper: https://arxiv.org/abs/1808.06670.
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
        Args:
            x_size: input shape for x, both the obs shape and the encoding shape are supported.
            y_size: input shape for y, both the obs shape and the encoding shape are supported.
            heads: a list of 2 int elems, heads[0] for x and head[1] for y.
                Used in multi-head, global-local, local-local MI maximization process.
            loss_type: only the InfoNCE loss is available now.
            temperature: the parameter to adjust the log_softmax.
        """
        super(ContrastiveLoss, self).__init__()
        assert len(heads) == 2, "Expected length of 2, but got: {}".format(len(heads))
        assert loss_type.lower() in ["infonce"]

        self._type = loss_type.lower()
        self._encode_shape = encode_shape
        self._heads = heads
        self._x_encoder = self._get_encoder(x_size, heads[0])
        self._y_encoder = self._get_encoder(y_size, heads[1])
        self._temperature = temperature

    def _get_encoder(self, obs: Union[int, SequenceType], heads: int):
        from ding.model import ConvEncoder, FCEncoder

        if isinstance(obs, int):
            obs = [obs]
        assert len(obs) in [1, 3]

        if len(obs) == 1:
            hidden_size_list = [128, 128, self._encode_shape * heads]
            encoder = FCEncoder(obs[0], hidden_size_list)
        else:
            hidden_size_list = [32, 64, 64, self._encode_shape * heads]
            if obs[-1] >= 36:
                encoder = ConvEncoder(obs, hidden_size_list)
            else:
                encoder = ConvEncoder(obs, hidden_size_list, kernel_size=[4, 3, 2], stride=[2, 1, 1])
        return encoder

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Computes the noise contrastive estimation-based loss, a.k.a. infoNCE.
        Args:
            x: the input x, both raw obs and encoding are supported.
            y: the input y, both raw obs and encoding are supported.
        Returns:
            torch.Tensor: loss value.
        """

        N = x.size(0)
        x_heads, y_heads = self._heads
        x = self._x_encoder.forward(x).view(N, x_heads, self._encode_shape)
        y = self._y_encoder.forward(y).view(N, y_heads, self._encode_shape)

        x_n = x.view(-1, self._encode_shape)
        y_n = y.view(-1, self._encode_shape)

        # Use inner product to obtain postive samples.
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

        # Concatenate postive and negative samples and apply log softmax.
        pred_lgt = torch.cat([u_pos, u_neg], dim=2)
        pred_log = F.log_softmax(pred_lgt * self._temperature, dim=2)

        # The positive score is the first element of the log softmax.
        loss = -pred_log[:, :, 0, :].mean()
        return loss
