from typing import Union
from ding.utils import SequenceType
import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.model import ConvEncoder, FCEncoder


class ContrastiveLoss(nn.Module):
    """
        The class for contrastive learning losses.
        Only InfoNCE loss supported currently.
        Code Reference: https://github.com/rdevon/DIM.
        paper: https://arxiv.org/abs/1808.06670.
    """

    def __init__(
            self,
            x_dim: Union[int, SequenceType],
            y_dim: Union[int, SequenceType],
            heads: SequenceType = [1, 1],
            encode_shape: int = 64,
            loss_type: str = "infonce",
            temperature: float = 1.0,
    ) -> None:
        """
        Args:
            x_dim: input dimensions for x, both the obs shape and the encoding shape are supported.
            y_dim: input dimensions for y, both the obs shape and the encoding shape are supported.
            heads: a list of 2 int elems, heads[0] for x and head[1] for y.
                Used in multi-head, global-local, local-local MI maximization process.
            loss_type: only InfoNCE loss supported currently.
            temperature: the parameter to adjust the log_softmax.
        """
        super(ContrastiveLoss, self).__init__()
        assert len(heads) == 2
        assert loss_type.lower() in ["infonce"]

        self._type = loss_type.lower()
        self._encode_shape = encode_shape
        self._heads = heads
        self._x_encoder = self._get_encoder(x_dim, heads[0])
        self._y_encoder = self._get_encoder(y_dim, heads[1])
        self._temperature = temperature

    def _get_encoder(self, obs: Union[int, SequenceType], heads: int):
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
        '''
        Computes the noise contrastive estimation-based loss, a.k.a. infoNCE.
        Args:
            x: the input x, both raw obs and encoding are supported.
            y: the input y, both raw obs and encoding are supported.
        Returns:
            torch.Tensor: loss value.
        '''

        N = x.size(0)
        x_heads, y_heads = self._heads
        x = self._x_encoder.forward(x).reshape(N, x_heads, self._encode_shape)
        y = self._y_encoder.forward(y).reshape(N, y_heads, self._encode_shape)

        x_n = x.reshape(-1, self._encode_shape)
        y_n = y.reshape(-1, self._encode_shape)

        # Inner product for positive samples. Outer product for negative. We need to do it this way
        # for the multiclass loss. For the outer product, we want a N x N x n_local x n_multi tensor.
        u_pos = torch.matmul(x, y.permute(0, 2, 1)).unsqueeze(2)
        u_all = torch.mm(y_n, x_n.t()).reshape(N, y_heads, N, x_heads).permute(0, 2, 3, 1)

        # We need to mask the diagonal part of the negative tensor.
        mask = torch.eye(N)[:, :, None, None].to(x.device)
        n_mask = 1 - mask

        # Masking is done by shifting the diagonal before exp.
        u_neg = (n_mask * u_all) - (10. * (1 - n_mask))  # mask out "self" examples, all diagonals are set to -10.
        u_neg = u_neg.reshape(N, N * x_heads, y_heads).unsqueeze(dim=1).expand(-1, x_heads, -1, -1)

        # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
        pred_lgt = torch.cat([u_pos, u_neg], dim=2)
        pred_log = F.log_softmax(pred_lgt * self._temperature, dim=2)

        # The positive score is the first element of the log softmax.
        loss = -pred_log[:, :, 0].mean()
        return loss

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        with torch.no_grad():
            out = self.forward(x, y)
        return out
