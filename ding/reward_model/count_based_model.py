import torch
from torch import distributions
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import random
from math import log, exp, pow

import numpy as np
from ding.utils import REWARD_MODEL_REGISTRY
from .base_reward_model import BaseRewardModel


@REWARD_MODEL_REGISTRY.register('countbased')
class CountbasedRewardModel(BaseRewardModel):
    """
        
    """
    
    config = dict(

        type='countbased',
        counter_type='PixelCNN',
        intrinsic_reward_type='add',
        bonus_coeffient=0.1,
        img_height=42,
        img_width=42,
        in_channels=1,
        out_channels=1,
        n_gated=2,
        gated_channels=16,
        head_channels=64,
        q_level=256,
    )

    def __init__(
        self,
        cfg: dict,
        device,
        tb_logger: 'SummaryWriter'
    ) -> None:  # noqa
        """
            # TODO:
        """
        super(BaseRewardModel, self).__init__()
        self.cfg: dict = cfg
        self._beta = cfg.bonus_coeffient
        self._counter_type = cfg.counter_type
        self.device = device
        self.tb_logger = tb_logger
        assert self._counter_type in ['SimHash', 'AutoEncoder', 'PixelCNN']
        # TODO: copy
        if self._counter_type == 'PixelCNN':
            print(cfg)
            self._counter = GatedPixelCNN(
                in_channels=cfg.in_channels,
                out_channels=cfg.out_channels,
                n_gated=cfg.n_gated,
                gated_channels=cfg.gated_channels,
                head_channels=cfg.head_channels,
                q_level=cfg.q_level,
            )
            self._counter.to(self.device)
            self.intrinsic_reward_type = cfg.intrinsic_reward_type
            self.obs_shape = (cfg.img_height, cfg.img_width)
            self.index_range = np.arange(cfg.img_height * cfg.img_width)
            assert self.intrinsic_reward_type in ['add', 'new', 'assign']
            self.train_data = []
            self.opt = optim.RMSprop(
                self._counter.parameters(),
                momentum=0.9,
                weight_decay=0.95,
                eps=1e-4,
            )

    def _train(self, train_data):
        '''
            Using sequence data to train the network.(1, H, W, 1)
        '''
        flattened_logits, target_pixel_loss, _ = self._counter(train_data)
        # flattened_logits:[BHWC, D]; target_pixel_loss:[BHWC]
        loss = nn.CrossEntropyLoss(reduction='none')(   # loss:[D]
            flattened_logits, target_pixel_loss.long()
        )
        print(loss.shape)
        loss = loss.mean()  # loss: [1]
        print(loss.shape)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self) -> None:
        pass

    def estimate(self, data, t) -> None:
        """
            # TODO:
        """
        if self._counter_type == 'PixelCNN':
            obs = self._collect_states(data)
            for o, item in zip(obs, data):
                o = o.unsqueeze(0).to(self.device)

                prob = (self._probs(o) + 1e-8).sum().item()
                
                self._train(o)

                with torch.no_grad():
                    recoding_prob = (self._probs(o) + 1e-8).sum().item()

                pred_gain = max(0, np.log(recoding_prob) - np.log(prob))

                intrinsic_reward = pow(
                    exp(self.cfg.bonus_coeffient * pow(t+1, -0.5) * pred_gain) - 1, 0.5
                )

                print(f'time:{t}, intrisic reward:{intrinsic_reward}')

                if self.intrinsic_reward_type == 'add':
                    item['reward'] += intrinsic_reward
                elif self.intrinsic_reward_type == 'new':
                    item['intrinsic_reward'] = intrinsic_reward
                elif self.intrinsic_reward_type == 'assign':
                    item['reward'] = intrinsic_reward

    def _probs(self, obs):
        _, indexes, target = self._counter(obs)
        pred_prob = target[self.index_range, indexes.long()]
        return pred_prob

    def _collect_states(self, data):
        obs = []
        for item in data:
            state = item['obs']
            state = state.unsqueeze(0)
            state = nn.functional.interpolate(state, 42)
            state = state.squeeze(0)
            obs.append(state)
        return obs
        
    def collect_data(self, data: list) -> None:
        pass

    def clear_data(self) -> None:
        pass


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
        # TODO(eugenhotaj): Is it better to shift down the the vstack_Nx1 output
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
        """Computes the forward pass.
        Args:
            vstack_input: The iin_channels=1,
        out_channels=1,
        n_gated=2,
        gated_channels=16,
        head_channels=64,
        q_level=256,
        """
        _, _, h, w = vstack_input.shape  # Assuming NCHW.

        # Compute vertical stack.
        vstack = self._vstack_Nx1(self._vstack_1xN(vstack_input))[:, :, :h, :]
        link = self._link(vstack)
        vstack += self._vstack_1x1(vstack_input)
        vstack = self._activation(vstack)

        # Compute horizontal stack.
        hstack = link + self._hstack_1xN(hstack_input)[:, :, :, :w]
        hstack = self._activation(hstack)
        skip = self._hstack_skip(hstack)
        hstack = self._hstack_residual(hstack)
        # We cannot use a residual connection for causal layers
        # otherwise we'll have access to future pixels.
        if not self._mask_center:
            hstack += hstack_input

        return vstack, hstack, skip


class GatedPixelCNN(nn.Module):
    """The Gated PixelCNN model."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        n_gated=2,
        gated_channels=16,
        head_channels=64,
        q_level=256,
    ):
        """Initializes a new GatedPixelCNN instance.
        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            n_gated: The number of gated layers (not including the input layers).
            gated_channels: The number of channels to use in the gated layers.
            head_channels: The number of channels to use in the 1x1 convolution blocks
                in the head after all the gated channels.
            sample_fn: See the base class.
        """
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
        self.q_level = q_level

    def forward(self, x):
        vstack, hstack, skip_connections = self._input(x, x)
        for gated_layer in self._gated_layers:
            vstack, hstack, skip = gated_layer(vstack, hstack)
            skip_connections += skip
        logits = self._head(skip_connections)

        logits, x = logits.permute(0, 2, 3, 1), x.permute(0, 2, 3, 1)

        flattened_logits = torch.reshape(logits, [-1, self.q_level])
        
        target_pixel_loss = torch.reshape(x, [-1])
        
        flattened_output = nn.Softmax(dim=-1)(flattened_logits)
        
        return flattened_logits, target_pixel_loss, flattened_output