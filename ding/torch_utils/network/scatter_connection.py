import torch
import torch.nn as nn
from typing import Tuple, List
from ding.hpc_rl import hpc_wrapper


def shape_fn_scatter_connection(args, kwargs) -> List[int]:
    """
    Overview:
        Return the shape of scatter_connection for HPC.
    Arguments:
        - args (:obj:`Tuple`): The arguments passed to the scatter_connection function.
        - kwargs (:obj:`Dict`): The keyword arguments passed to the scatter_connection function.
    Returns:
        - shape (:obj:`List[int]`): A list representing the shape of scatter_connection, \
            in the form of [B, M, N, H, W, scatter_type].
    """
    if len(args) <= 1:
        tmp = list(kwargs['x'].shape)
    else:
        tmp = list(args[1].shape)  # args[0] is __main__.ScatterConnection object
    if len(args) <= 2:
        tmp.extend(kwargs['spatial_size'])
    else:
        tmp.extend(args[2])
    tmp.append(args[0].scatter_type)
    return tmp


class ScatterConnection(nn.Module):
    """
    Overview:
        Scatter feature to its corresponding location. In AlphaStar, each entity is embedded into a tensor,
        and these tensors are scattered into a feature map with map size.
    Interfaces:
        ``__init__``, ``forward``, ``xy_forward``
    """

    def __init__(self, scatter_type: str) -> None:
        """
        Overview:
            Initialize the ScatterConnection object.
        Arguments:
            - scatter_type (:obj:`str`): The scatter type, which decides the behavior when two entities have the \
                same location. It can be either 'add' or 'cover'. If 'add', the first one will be added to the \
                second one. If 'cover', the first one will be covered by the second one.
        """
        super(ScatterConnection, self).__init__()
        self.scatter_type = scatter_type
        assert self.scatter_type in ['cover', 'add']

    @hpc_wrapper(
        shape_fn=shape_fn_scatter_connection,
        namedtuple_data=False,
        include_args=[0, 2],
        include_kwargs=['x', 'location'],
        is_cls_method=True
    )
    def forward(self, x: torch.Tensor, spatial_size: Tuple[int, int], location: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Scatter input tensor 'x' into a spatial feature map.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor of shape `(B, M, N)`, where `B` is the batch size, `M` \
                is the number of entities, and `N` is the dimension of entity attributes.
            - spatial_size (:obj:`Tuple[int, int]`): The size `(H, W)` of the spatial feature map into which 'x' \
                will be scattered, where `H` is the height and `W` is the width.
            - location (:obj:`torch.Tensor`): The tensor of locations of shape `(B, M, 2)`. \
                Each location should be (y, x).
        Returns:
            - output (:obj:`torch.Tensor`): The scattered feature map of shape `(B, N, H, W)`.
        Note:
            When there are some overlapping in locations, 'cover' mode will result in the loss of information.
            'add' mode is used as a temporary substitute.
        """
        device = x.device
        B, M, N = x.shape
        x = x.permute(0, 2, 1)
        H, W = spatial_size
        index = location[:, :, 1] + location[:, :, 0] * W
        index = index.unsqueeze(dim=1).repeat(1, N, 1)
        output = torch.zeros(size=(B, N, H, W), device=device).view(B, N, H * W)
        if self.scatter_type == 'cover':
            output.scatter_(dim=2, index=index, src=x)
        elif self.scatter_type == 'add':
            output.scatter_add_(dim=2, index=index, src=x)
        output = output.view(B, N, H, W)
        return output

    def xy_forward(
            self, x: torch.Tensor, spatial_size: Tuple[int, int], coord_x: torch.Tensor, coord_y
    ) -> torch.Tensor:
        """
        Overview:
            Scatter input tensor 'x' into a spatial feature map using separate x and y coordinates.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor of shape `(B, M, N)`, where `B` is the batch size, `M` \
                is the number of entities, and `N` is the dimension of entity attributes.
            - spatial_size (:obj:`Tuple[int, int]`): The size `(H, W)` of the spatial feature map into which 'x' \
                will be scattered, where `H` is the height and `W` is the width.
            - coord_x (:obj:`torch.Tensor`): The x-coordinates tensor of shape `(B, M)`.
            - coord_y (:obj:`torch.Tensor`): The y-coordinates tensor of shape `(B, M)`.
        Returns:
            - output (:obj:`torch.Tensor`): The scattered feature map of shape `(B, N, H, W)`.
        Note:
            When there are some overlapping in locations, 'cover' mode will result in the loss of information.
            'add' mode is used as a temporary substitute.
        """
        device = x.device
        B, M, N = x.shape
        x = x.permute(0, 2, 1)
        H, W = spatial_size
        index = (coord_x * W + coord_y).long()
        index = index.unsqueeze(dim=1).repeat(1, N, 1)
        output = torch.zeros(size=(B, N, H, W), device=device).view(B, N, H * W)
        if self.scatter_type == 'cover':
            output.scatter_(dim=2, index=index, src=x)
        elif self.scatter_type == 'add':
            output.scatter_add_(dim=2, index=index, src=x)
        output = output.view(B, N, H, W)
        return output
