import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftArgmax(nn.Module):
    r"""
    Overview:
        An nn.Module that computes SoftArgmax
    Interface:
        __init__, forward

    .. note:

        For more softargmax info, you can refere to
        the wiki page <https://wikimili.com/en/Softmax_function> or
        the lecture <https://mc.ai/softmax-function-beyond-the-basics/>

    """

    def __init__(self):
        r"""
        Overview:
            Initialize the SoftArgmax module
        """
        super(SoftArgmax, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Soft-argmax for location regression
        Arguments:
            - x (:obj:`torch.Tensor`): predict heat map
        Returns:
            - location (:obj:`torch.Tensor`): predict location
        Shapes:
            - x: :math:`(B, C, H, W)`, while B is the batch size, C is number of channels, \
                H and W stands for height and width
            - location: :math:`(B, 2)`, while B is the batch size
        """
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        # 1 channel
        assert (x.shape[1] == 1)
        h_kernel = torch.arange(0, H, device=device).to(dtype)
        h_kernel = h_kernel.view(1, 1, H, 1).repeat(1, 1, 1, W)
        w_kernel = torch.arange(0, W, device=device).to(dtype)
        w_kernel = w_kernel.view(1, 1, 1, W).repeat(1, 1, H, 1)
        x = F.softmax(x.view(B, C, -1), dim=-1).view(B, C, H, W)
        h = (x * h_kernel).sum(dim=[1, 2, 3])
        w = (x * w_kernel).sum(dim=[1, 2, 3])
        return torch.stack([h, w], dim=1)
