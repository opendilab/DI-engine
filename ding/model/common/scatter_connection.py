from typing import Tuple
import torch
import torch.nn as nn


class ScatterConnection(nn.Module):
    """
    Overview:
        Scatter feature to its corresponding location
        In alphastar, each entity is embedded into a tensor, these tensors are scattered into a feature map with map size
    """

    def __init__(self, scatter_type='add') -> None:
        """
        Overview:
            Init class
        Arguments:
            - scatter_type (:obj:`str`): add or cover, if two entities have same location, scatter type decides the
                first one should be covered or added to second one
        """
        super(ScatterConnection, self).__init__()
        self.scatter_type = scatter_type
        assert self.scatter_type in ['cover', 'add']

    def xy_forward(
            self, x: torch.Tensor, spatial_size: Tuple[int, int], coord_x: torch.Tensor, coord_y
    ) -> torch.Tensor:
        """
        Overview:
            scatter x into a spatial feature map
        Arguments:
            - x (:obj:`tensor`): input tensor :math: `(B, M, N)` where `M` means the number of entity, `N` means\
                the dimension of entity attributes
            - spatial_size (:obj:`tuple`): Tuple[H, W], the size of spatial feature x will be scattered into
            - coord_x (:obj:`tensor`): :math: `(B, M)` torch.LongTensor, each location should be x
            - coord_y (:obj:`tensor`): :math: `(B, M)` torch.LongTensor, each location should be y
        Returns:
            - output (:obj:`tensor`): :math: `(B, N, H, W)` where `H` and `W` are spatial_size, return the\
                scattered feature map
        Shapes:
            - Input: :math: `(B, M, N)` where `M` means the number of entity, `N` means\
                the dimension of entity attributes
            - Size: Tuple[H, W]
            - Coord_x: :math: `(B, M)` torch.LongTensor, each location should be x
            - Coord_y: :math: `(B, M)` torch.LongTensor, each location should be y
            - Output: :math: `(B, N, H, W)` where `H` and `W` are spatial_size

        note:
            when there are some overlapping in locations, ``cover`` mode will result in the loss of information, we
            use the addition as temporal substitute.
        """
        device = x.device
        BatchSize, Num, EmbeddingSize = x.shape
        x = x.permute(0, 2, 1)
        H, W = spatial_size
        indices = (coord_x * W + coord_y).long()
        indices = indices.unsqueeze(dim=1).repeat(1, EmbeddingSize, 1)
        output = torch.zeros(size=(BatchSize, EmbeddingSize, H, W), device=device).view(BatchSize, EmbeddingSize, H * W)
        if self.scatter_type == 'cover':
            output.scatter_(dim=2, index=indices, src=x)
        elif self.scatter_type == 'add':
            output.scatter_add_(dim=2, index=indices, src=x)
        output = output.view(BatchSize, EmbeddingSize, H, W)
        return output

    def forward(self, x: torch.Tensor, spatial_size: Tuple[int, int], location: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            scatter x into a spatial feature map
        Arguments:
            - x (:obj:`tensor`): input tensor :math: `(B, M, N)` where `M` means the number of entity, `N` means\
                the dimension of entity attributes
            - spatial_size (:obj:`tuple`): Tuple[H, W], the size of spatial feature x will be scattered into
            - location (:obj:`tensor`): :math: `(B, M, 2)` torch.LongTensor, each location should be (y, x)
        Returns:
            - output (:obj:`tensor`): :math: `(B, N, H, W)` where `H` and `W` are spatial_size, return the\
                scattered feature map
        Shapes:
            - Input: :math: `(B, M, N)` where `M` means the number of entity, `N` means\
                the dimension of entity attributes
            - Size: Tuple[H, W]
            - Location: :math: `(B, M, 2)` torch.LongTensor, each location should be (y, x)
            - Output: :math: `(B, N, H, W)` where `H` and `W` are spatial_size

        note:
            when there are some overlapping in locations, ``cover`` mode will result in the loss of information, we
            use the addition as temporal substitute.
        """
        device = x.device
        BatchSize, Num, EmbeddingSize = x.shape
        x = x.permute(0, 2, 1)
        H, W = spatial_size
        indices = location[:, :, 1] + location[:, :, 0] * W
        indices = indices.unsqueeze(dim=1).repeat(1, EmbeddingSize, 1)
        output = torch.zeros(size=(BatchSize, EmbeddingSize, H, W), device=device).view(BatchSize, EmbeddingSize, H * W)
        if self.scatter_type == 'cover':
            output.scatter_(dim=2, index=indices, src=x)
        elif self.scatter_type == 'add':
            output.scatter_add_(dim=2, index=indices, src=x)
        output = output.view(BatchSize, EmbeddingSize, H, W)
        return output