import torch
from torch import distributions
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import random
from math import log, exp, pow
from torchvision.utils import save_image
import numpy as np

'''
Code implementation reference:
    [1] https://github.com/NoListen/ERL
    [2] https://github.com/EugenHotaj/pytorch-generative
'''

class GatedActivation(nn.Module):
    """Gated activation function as introduced in https://arxiv.org/pdf/1703.01310.pdf.
    The function computes actiation_fn(f) * sigmoid(g). The f and g correspond to the
    top 1/2 and bottom 1/2 of the input channels.
    """

    def __init__(self, activation_fn=torch.nn.Tanh()):
        """Initializes a new GatedActivation instance.
        Args:
            activation_fn: Activation to use for the top 1/2 input channels.
        """
        super().__init__()
        self._activation_fn = activation_fn

    def forward(self, x):
        _, c, _, _ = x.shape
        assert c % 2 == 0, "x must have an even number of channels."
        x, gate = x[:, : c // 2, :, :], x[:, c // 2 :, :, :]
        return self._activation_fn(x) * torch.sigmoid(gate)


class GatedPixelCNNLayer(nn.Module):
    """A Gated PixelCNN layer.
    The layer takes as input 'vstack' and 'hstack' from previous
    'GatedPixelCNNLayers' and returns 'vstack', 'hstack', 'skip' where 'skip' is
    the skip connection to the pre-logits layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, mask_center=False):
        """Initializes a new GatedPixelCNNLayer instance.
        Args:
            in_channels: The number of channels in the input.
            out_channels: The number of output channels.
            kernel_size: The size of the (causal) convolutional kernel to use.
            mask_center: Whether the 'GatedPixelCNNLayer' is causal. If 'True', the
                center pixel is masked out so the computation only depends on pixels to
                the left and above. The residual connection in the horizontal stack is
                also removed.
        """
        super().__init__()

        assert kernel_size % 2 == 1, "kernel_size cannot be even"

        self._in_channels = in_channels
        self._out_channels = out_channels
        # change
        self._activation = GatedActivation()
        # self._activation = pg_nn.GatedActivation()
        self._kernel_size = kernel_size
        self._padding = (kernel_size - 1) // 2  # (kernel_size - stride) / 2
        self._mask_center = mask_center

        # Vertical stack convolutions.
        self._vstack_1xN = nn.Conv2d(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            kernel_size=(1, self._kernel_size),
            padding=(0, self._padding),
        )
        # instead of adding extra padding to the convolution? When we add extra
        # padding, the cropped output rows will no longer line up with the rows of
        # the vstack_1x1 output.
        self._vstack_Nx1 = nn.Conv2d(
            in_channels=self._out_channels,
            out_channels=2 * self._out_channels,
            kernel_size=(self._kernel_size // 2 + 1, 1),
            padding=(self._padding + 1, 0),
        )
        self._vstack_1x1 = nn.Conv2d(
            in_channels=in_channels, out_channels=2 * out_channels, kernel_size=1
        )

        self._link = nn.Conv2d(
            in_channels=2 * out_channels, out_channels=2 * out_channels, kernel_size=1
        )

        # Horizontal stack convolutions.
        self._hstack_1xN = nn.Conv2d(
            in_channels=self._in_channels,
            out_channels=2 * self._out_channels,
            kernel_size=(1, self._kernel_size // 2 + 1),
            padding=(0, self._padding + int(self._mask_center)),
        )
        self._hstack_residual = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1
        )
        self._hstack_skip = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, vstack_input, hstack_input):
        r"""
        Overview:
            PixelCNNLayer forward computation graph, input observation tensor to predict features bedore current pixel.
        Arguments:
            - vstack_input (:obj:`torch.Tensor`): Observation inputs for vertical convolution.
            - hstack_input (:obj:`torch.Tensor`): Observation inputs for horizontal convolution.
        Returns:
            - vstack (:obj:`torch.Tensor`): result after vertical convolution.
            - hstack (:obj:`torch.Tensor`): result after horizonal convolution.
            - skip (:onj:`torch.Tensor`): the skip connection to the pre-logits layer.
        Shapes:
            - vstack_input (:obj:`torch.Tensor`): :math:`(B, C, H, W)`.
            - vstack (:obj:`torch.Tensor`): :math:`(B, gated_channels(defaultL:16), H, W)`.
            - hstack (:obj:`torch.Tensor`): :math:`(B, gated_channels, H, W)`.
            - skip (:obj:`torch.Tensor`): :math:`(B, gated_channels(defaultL:16), H, W)`.
        Examples:
            >>> model = GatedPixelCNNLayer()     # default parameters: B=1, C=1, H=W=42, D=256.
            >>> vstack_input, hstack_input = torch.rand(1, 1, 42, 42), torch.rand(1, 1, 42, 42).
            >>> vstack, hstack, skip = model(vstack_input, hstack_input)
            >>> assert isinstance(flattened_logits, torch.Tensor) and vstack.shape == torch.Size([1, 16, 42, 42])
            >>> assert isinstance(target_pixel_loss, torch.Tensor) and hstack.shape == torch.Size([1, 16, 42, 42])
            >>> assert isinstance(flattened_output, torch.Tensor) and skip.shape == torch.Size([1, 16, 42, 42])
        """
        _, _, h, w = vstack_input.shape  # Assuming BCHW.

        # Compute vertical stack.
        vstack = self._vstack_Nx1(self._vstack_1xN(vstack_input))[:, :, :h, :]
        link = self._link(vstack)
        vstack += self._vstack_1x1(vstack_input)
        vstack = self._activation(vstack)

        # Compute horizontal stack.
        hstack = link + self._hstack_1xN(hstack_input)[..., :w]
        hstack = self._activation(hstack)
        skip = self._hstack_skip(hstack)
        hstack = self._hstack_residual(hstack)
        # We cannot use a residual connection for causal layers
        # otherwise we'll have access to future pixels.
        if not self._mask_center:
            hstack += hstack_input


        return vstack, hstack, skip


class GatedPixelCNN(nn.Module):
    """
    The Gated PixelCNN model.
    
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        n_gated=2,
        gated_channels=16,
        head_channels=64,
        q_level=8,
    ):
        """Initializes a new GatedPixelCNN instance.
        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            n_gated: The number of gated layers (not including the input layers).
            gated_channels: The number of channels to use in the gated layers.
            head_channels: The number of channels to use in the 1x1 convolution blocks
                in the head after all the gated channels.
            q_level: The number of levels to quantisize value of each channel of each pixel into
        """
        assert in_channels == out_channels
        super().__init__()
        self._input = GatedPixelCNNLayer(
            in_channels=in_channels,
            out_channels=gated_channels,
            kernel_size=7,
            mask_center=True,
        )
        self._gated_layers = nn.ModuleList(
            [
                GatedPixelCNNLayer(
                    in_channels=gated_channels,
                    out_channels=gated_channels,
                    # paper in exploration is 1, default pixelcnn paper is 3
                    kernel_size=1,
                    mask_center=False,
                )
                for _ in range(n_gated)
            ]
        )
        self._head = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=gated_channels, out_channels=head_channels, kernel_size=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=head_channels, out_channels=q_level * out_channels, kernel_size=1
            ),
        )
        self.num_channels = in_channels
        self.q_level = q_level

    def forward(self, x):
        r"""
        Overview:
            PixelCNN forward computation graph, input observation tensor to predict pseudo-count.
        Arguments:
            - x (:obj:`torch.Tensor`): Observation inputs
        Returns:
            - flattened_logits (:obj:'torch.Tensor'): reshaped logits.
            - target_pixel_loss (:obj:'torch.Tensor'): target pixel which comes from input x.
            - flattened_output (:onj:'torch.Tensor'): probability distribution of flattened_logits with a softmax function.
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C, H, W)`.
            - flattened_logit (:obj:`torch.Tensor`): :math:`(B*H*W*C, D)`.
            - target_pixel_loss (:obj:`torch.Tensor`): :math:`(B*H*W*C)`.
            - flattened_output (:obj:`torcg.Tensor`): :math:`(B*H*W*C, D)`. 
        Examples:
            >>> model = GatedPixelCNN()     # default parameters: B=1, C=1, H=W=42, D=256.
            >>> inputs = torch.rand(1, 1, 42, 42)
            >>> flattened_logits, target_pixel_loss, flattened_output = model(inputs)
            >>> assert isinstance(flattened_logits, torch.Tensor) and flattened_logits.shape == torch.Size([1764, 256])
            >>> assert isinstance(target_pixel_loss, torch.Tensor) and target_pixel_loss.shape == torch.Size([1764])
            >>> assert isinstance(flattened_output, torch.Tensor) and flattened_output.shape == torch.Size([1764, 256])
        """
        x_ = x.type(torch.float32)
        # shape [B, C, H, W]
        vstack, hstack, skip_connections = self._input(x_, x_)
        
        # shape [B, gated_channels(default:16), H, W]
        for gated_layer in self._gated_layers:
            vstack, hstack, skip = gated_layer(vstack, hstack)
            skip_connections += skip
        
        # shape [B, DC, H, W]
        logits = self._head(skip_connections)
        # print(f'logits:{logits.shape}')


        # shape [B, DC, H, W] -> [B, H, W, DC]
        logits, x = logits.permute(0, 2, 3, 1), x.permute(0, 2, 3, 1)

        if self.num_channels > 1:
            # shape [B, H, W, DC] -> [B, H, W, D, C]
            logits = torch.reshape(logits, [-1, 42, 42, self.q_level, self.num_channels])
            
            # shape [B, H, W, D, C] -> [B, H, W, C, D]
            logits = logits.permute(0, 1, 2 ,4, 3)

        # shape [B, H, W, DC] -> [BHWC, D]
        flattened_logits = torch.reshape(logits, [-1, self.q_level])
        
        # shape [B, H, W, C] -> [BHWC]
        target_pixel_loss = torch.reshape(x, [-1])
        
        # shape [BHWC, D], values [probability distribution]
        flattened_output = nn.Softmax(dim=-1)(flattened_logits)
        
        return flattened_logits, target_pixel_loss, flattened_output