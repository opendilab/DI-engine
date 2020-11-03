import torch
import torch.nn as nn
from typing import Tuple


class ScatterConnection(nn.Module):

    def __init__(self) -> None:
        super(ScatterConnection, self).__init__()

    def forward(self, x: torch.Tensor, spatial_size: Tuple[int, int], location: torch.Tensor) -> torch.Tensor:
        """
        Shape:
            - Input: :math: `(B, M, N)` where `M` means the number of entity, `N` means
              the dimension of entity attributes
            - Size: Tuple[H, W]
            - Location: :math: `(B, M, 2)` torch.LongTensor, each location should be (y, x)
            - Output: :math: `(B, N, H, W)` where `H` and `W` are spatial_size
        Note:
            location must be not overlapped
        """
        device = x.device
        B, M, N = x.shape
        H, W = spatial_size
        output = torch.zeros(B, N, H * W, device=device)
        for b in range(B):
            index = location[b, :, 0] * W + location[b, :, 1]
            index = index.unsqueeze(0).repeat(N, 1)
            src = x[b].permute(1, 0)
            output[b].scatter_(dim=1, index=index, src=src)
        output = output.reshape(B, N, H, W)
        return output
