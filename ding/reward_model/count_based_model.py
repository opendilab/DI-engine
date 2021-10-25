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
    Overview:
        The Count based reward model class
    Interface:
        ``estimate``, ``train``, ``load_expert_data``, ``collect_data``, ``clear_date``, \
            ``__init__``, ``_train``, ``_batch_mn_pdf``
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
        q_level=8,
    )

    def __init__(
        self,
        cfg: dict,
        device,
        tb_logger: 'SummaryWriter'
    ) -> None:  # noqa
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature.
            Some rules in naming the attributes of ``self.``:
                - ``e_`` : expert values
                - ``_sigma_`` : standard division values
                - ``p_`` : current policy values
                - ``_s_`` : states
                - ``_a_`` : actions
        Arguments:
            - cfg (:obj:`Dict`): Training config
            - device (:obj:`str`): Device usage, i.e. "cpu" or "cuda"
            - tb_logger (:obj:`str`): Logger, defaultly set as 'SummaryWriter' for model summary
        """
        super(BaseRewardModel, self).__init__()
        self.cfg: dict = cfg
        self._counter_type = cfg.counter_type
        self.device = device
        self.tb_logger = tb_logger
        assert self._counter_type in ['SimHash', 'AutoEncoder', 'PixelCNN']
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
            self.index_range = np.arange(cfg.in_channels * cfg.img_height * cfg.img_width)
            assert self.intrinsic_reward_type in ['add', 'new', 'assign']
            self.opt = optim.RMSprop(
                self._counter.parameters(),
                momentum=0.9,
                weight_decay=0.95,
                eps=1e-4,
            )

    def _execute_gain_training(self, train_data: torch.Tensor, train_iter: int):
        '''
        Overview:
            Using input data to train the network while estimating intrintic reward.
        Arguments:
            train_data (:obj:`torch.Tensor`): Observation with shape [1, 1, 42, 42].
        '''
        flattened_logits, target_pixel_loss, _ = self._counter(train_data)

        # flattened_logits shape: [BHWC, D]; target_pixel_loss shape: [BHWC]
        loss = nn.CrossEntropyLoss(reduction='none')(
            flattened_logits, target_pixel_loss
        ).mean()

        self.tb_logger.add_scalar('reward_model/reward_model_loss', loss, train_iter)
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self) -> None:
        """
        Overview:
            Training the PixelCNN reward model.
        """
        pass

    def estimate(self, data: list, train_iter: int) -> None:
        """
        Overview:
            Estimate reward by rewriting the reward keys.
        Arguments:
            - data (:obj:`list`): the list of data used for estimation,\
                 with at least ``obs`` and ``action`` keys.
        Effects:
            - This is a side effect function which updates the reward values in place.
        """
        if self._counter_type == 'PixelCNN':
            obs = self._collect_states(data)

            probs = self._get_pseudo_count(obs)

            self._execute_gain_training(obs, train_iter)

            with torch.no_grad():
                recoding_probs = self._get_pseudo_count(obs).detach()

            batch = obs.shape[0]
            reward_mean = 0
            pred_gain_mean = 0
            for i, item in enumerate(data):
                pred_gain = torch.sum(torch.log(recoding_probs[i] + 1e-8) - torch.log(probs[i] + 1e-8))
                pred_gain_mean += pred_gain

                intrinsic_reward = pow(
                    exp(self.cfg.bonus_coeffient * pow(train_iter + 1, -0.5) * max(0, pred_gain)) - 1, 0.5
                )
                reward_mean += intrinsic_reward
                # reward clip in the original paper
                if intrinsic_reward > 1.:
                    intrinsic_reward = 1.
                if intrinsic_reward < -1.:
                    intrinsic_reward = -1.

                if self.intrinsic_reward_type == 'add':
                    item['reward'] += intrinsic_reward
                elif self.intrinsic_reward_type == 'new':
                    item['intrinsic_reward'] = intrinsic_reward
                elif self.intrinsic_reward_type == 'assign':
                    item['reward'] = intrinsic_reward
            
            reward_mean /= batch
            self.tb_logger.add_scalar('reward_model/intrinsic_reward', reward_mean, train_iter)
            pred_gain_mean /= batch
            self.tb_logger.add_scalar('reward_model/pred_gain', pred_gain_mean, train_iter)

    def _get_pseudo_count(self, obs: torch.Tensor):
        '''
        Overview:
            Compute the pseudo-count of given obs.
        Arguments:
            obs (:obj:`torch.Tensor`): Observation with shape [1, 1, 42, 42].
        '''
        _, indexes, target = self._counter(obs)
        batch = obs.shape[0]
        indexes = torch.reshape(indexes, [batch, -1])

        pred_prob = [target[self.index_range, indexes[i]] + 1e-8 for i in range(batch)]

        return torch.stack(pred_prob)

    def _collect_states(self, data: list):
        '''
        Overview:
            Get item 'obs' from data and reshape obs to [1, 42, 42], where shape format is [C, H, W].
        Arguments:
            - data (:obj:`list`): Raw training data (e.g. som form of states actions, obs, etc)
        Effects:
            This is function to get item 'obs' from input data and reshape it to [1, 42, 42] where shape format is [C, H, W].
        '''
        obs = [item['obs'] for item in data]
        obs = torch.stack(obs).to(self.device)
        _, x, y, _ = obs.shape
        if x==y:    # HWC
            obs = obs.permute(0, 3, 1, 2)

        obs = nn.functional.interpolate(obs, 42)

        obs = obs.permute(0, 2, 3, 1)
        # quantize the input
        obs = torch.clip(
            ((obs * self.cfg.q_level).type(torch.int64)), 0, self.cfg.q_level - 1
        )
        obs = obs.permute(0, 3, 1, 2)

        return obs
        
    def collect_data(self, data: list) -> None:
        """
        Overview:
            Collecting training data by iterating data items in the input list
        Arguments:
            - data (:obj:`list`): Raw training data (e.g. some form of states, actions, obs, etc)
        Effects:
            - This is a side effect function which updates the data attribute in ``self`` by \
                iterating data items in the input data items' list
        """
        pass

    def clear_data(self) -> None:
        """
        Overview:
            Clearing training data. \
            This is a side effect function which clears the data attribute in ``self``
        """
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
    """The Gated PixelCNN model."""

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