import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftArgmax(nn.Module):
    """
    Overview:
        A neural network module that computes the SoftArgmax operation (essentially a 2-dimensional spatial softmax),
        which is often used for location regression tasks. It converts a feature map (such as a heatmap) into precise
        coordinate locations.
    Interfaces:
        ``__init__``, ``forward``

    .. note::
        For more information on SoftArgmax, you can refer to <https://en.wikipedia.org/wiki/Softmax_function>
        and the paper <https://arxiv.org/pdf/1504.00702.pdf>.
    """

    def __init__(self):
        """
        Overview:
            Initialize the SoftArgmax module.
        """
        super(SoftArgmax, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Perform the forward pass of the SoftArgmax operation.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor, typically a heatmap representing predicted locations.
        Returns:
            - location (:obj:`torch.Tensor`): The predicted coordinates as a result of the SoftArgmax operation.
        Shapes:
            - x: :math:`(B, C, H, W)`, where `B` is the batch size, `C` is the number of channels, \
                and `H` and `W` represent height and width respectively.
            - location: :math:`(B, 2)`, where `B` is the batch size and 2 represents the coordinates (height, width).
        """
        # Unpack the dimensions of the input tensor
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        # Ensure the input tensor has a single channel
        assert C == 1, "Input tensor should have only one channel"
        # Create a meshgrid for the height (h_kernel) and width (w_kernel)
        h_kernel = torch.arange(0, H, device=device).to(dtype)
        h_kernel = h_kernel.view(1, 1, H, 1).repeat(1, 1, 1, W)

        w_kernel = torch.arange(0, W, device=device).to(dtype)
        w_kernel = w_kernel.view(1, 1, 1, W).repeat(1, 1, H, 1)

        # Apply the softmax function across the spatial dimensions (height and width)
        x = F.softmax(x.view(B, C, -1), dim=-1).view(B, C, H, W)
        # Compute the expected values for height and width by multiplying the probability map by the meshgrids
        h = (x * h_kernel).sum(dim=[1, 2, 3])  # Sum over the channel, height, and width dimensions
        w = (x * w_kernel).sum(dim=[1, 2, 3])  # Sum over the channel, height, and width dimensions

        # Stack the height and width coordinates along a new dimension to form the final output tensor
        return torch.stack([h, w], dim=1)
