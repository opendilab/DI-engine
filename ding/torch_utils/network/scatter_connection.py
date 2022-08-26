import torch
import torch.nn as nn
from typing import Tuple
from ding.hpc_rl import hpc_wrapper


def shape_fn_scatter_connection(args, kwargs) -> list:
    r"""
    Overview:
        Return shape of scatter_connection for hpc
    Returns:
        - shape (:obj:`list`): List like [B, M, N, H, W, scatter_type]
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
    r"""
    Overview:
        Scatter feature to its corresponding location
        In AlphaStar, each entity is embedded into a tensor,
        and these tensors are scattered into a feature map with map size.
    """

    def __init__(self, scatter_type: str) -> None:
        r"""
        Overview:
            Init class
        Arguments:
            - scatter_type (:obj:`str`): Supports ['add', 'cover']. If two entities have the same location, \
                scatter_type decides the first one should be covered or added to second one
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
            scatter x into a spatial feature map
        Arguments:
            - x (:obj:`tensor`): input tensor :math: `(B, M, N)` where `M` means the number of entity, `N` means \
                the dimension of entity attributes
            - spatial_size (:obj:`tuple`): Tuple[H, W], the size of spatial feature x will be scattered into
            - location (:obj:`tensor`): :math: `(B, M, 2)` torch.LongTensor, each location should be (y, x)
        Returns:
            - output (:obj:`tensor`): :math: `(B, N, H, W)` where `H` and `W` are spatial_size, return the\
                scattered feature map
        Shapes:
            - Input: :math: `(B, M, N)` where `M` means the number of entity, `N` means \
                the dimension of entity attributes
            - Size: Tuple type :math: `[H, W]`
            - Location: :math: `(B, M, 2)` torch.LongTensor, each location should be (y, x)
            - Output: :math: `(B, N, H, W)` where `H` and `W` are spatial_size

        .. note::

            When there are some overlapping in locations, ``cover`` mode will result in the loss of information, we
            use the addition as temporal substitute.
        """
        device = x.device
        B, M, N = x.shape
        H, W = spatial_size
        index = location.view(-1, 2).long()
        bias = torch.arange(B).mul_(H * W).unsqueeze(1).repeat(1, M).view(-1).to(device)

        index = index[:, 0] * W + index[:, 1]
        index += bias
        index = index.repeat(N, 1)

        x = x.view(-1, N).permute(1, 0)
        output = torch.zeros(N, B * H * W, device=device)
        if self.scatter_type == 'cover':
            output.scatter_(dim=1, index=index, src=x)
        elif self.scatter_type == 'add':
            output.scatter_add_(dim=1, index=index, src=x)
        output = output.reshape(N, B, H, W)
        output = output.permute(1, 0, 2, 3).contiguous()
        return output


def scatter_connection_v2(shape, project_embeddings, entity_location, scatter_dim, scatter_type='add'):
    B, H, W = shape
    device = entity_location.device
    entity_num = entity_location.shape[1]
    index = entity_location.view(-1, 2).long()
    bias = torch.arange(B).unsqueeze(1).repeat(1, entity_num).view(-1).to(device)
    bias *= H * W
    index[:, 0].clamp_(0, W - 1)
    index[:, 1].clamp_(0, H - 1)
    index = index[:, 1] * W + index[:, 0]  # entity_location: (x, y), spatial_info: (y, x)
    index += bias
    index = index.repeat(scatter_dim, 1)
    # flat scatter map and project embeddings
    scatter_map = torch.zeros(scatter_dim, B * H * W, device=device)
    project_embeddings = project_embeddings.view(-1, scatter_dim).permute(1, 0)
    if scatter_type == 'cover':
        scatter_map.scatter_(dim=1, index=index, src=project_embeddings)
    elif scatter_type == 'add':
        scatter_map.scatter_add_(dim=1, index=index, src=project_embeddings)
    else:
        raise NotImplementedError
    scatter_map = scatter_map.reshape(scatter_dim, B, H, W)
    scatter_map = scatter_map.permute(1, 0, 2, 3)
    return scatter_map
