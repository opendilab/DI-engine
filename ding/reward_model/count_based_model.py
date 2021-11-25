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
from ding.utils import REWARD_MODEL_REGISTRY
from .base_reward_model import BaseRewardModel
from ding.torch_utils.network import GatedPixelCNN

@REWARD_MODEL_REGISTRY.register('countbased')
class CountbasedRewardModel(BaseRewardModel):
    """
    Overview:
        The Count based reward model class
    Interface:
        ``estimate``, ``train``, ``load_expert_data``, ``collect_data``, ``clear_date``, \
            ``__init__``, ``_train``, ``_batch_mn_pdf``
    Paper implementation reference:
        [1] Count-based exploration with neural density models[C]
        Ostrovski G, Bellemare M G, Oord A, et al.
        International conference on machine learning. PMLR, 2017: 2721-2730.
        https://arxiv.org/abs/1703.01310
        [2] Conditional image generation with pixelcnn decoders[J].
        Oord A, Kalchbrenner N, Vinyals O, et al.
        arXiv preprint arXiv:1606.05328, 2016.
        https://arxiv.org/abs/1606.05328
    Code implementation reference:
        [1] https://github.com/NoListen/ERL
        [2] https://github.com/EugenHotaj/pytorch-generative
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
        bonus_scale=0.1,
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
            self.bonus_scale = cfg.bonus_scale
            print(self._counter)
        self.step = 0

    def _execute_gain_training(self, train_data: torch.Tensor, train_step: int):
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

        # recover
        # img_gen = torch.reshape(flattened_logits, [-1, 42, 42, 3])


        self.tb_logger.add_scalar('reward_model/reward_model_loss', loss, train_step)
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self) -> None:
        """
        Overview:
            Training the PixelCNN reward model.
        """
        pass

    def estimate(self, data: list) -> None:
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

            obses = self._collect_states(data)

            for obs, transition in zip(obses, data):

                obs = obs.unsqueeze(0)

                probs = self._get_pseudo_count(obs)
                
                self._execute_gain_training(obs, self.step)

                with torch.no_grad():
                    recoding_probs = self._get_pseudo_count(obs).detach()

                pred_gain = torch.sum(torch.log(recoding_probs[0] + 1e-8) - torch.log(probs[0] + 1e-8))

                intrinsic_reward = pow(
                    (exp(self.cfg.bonus_coeffient * pow(transition['step'] + 1, -0.5) * max(0, pred_gain)) - 1), 0.5
                )
                intrinsic_reward *= self.cfg.bonus_scale
    
                if self.intrinsic_reward_type == 'add':
                    transition['reward'][0] += intrinsic_reward
                elif self.intrinsic_reward_type == 'new':
                    transition['intrinsic_reward'][0] = intrinsic_reward
                elif self.intrinsic_reward_type == 'assign':
                    transition['reward'][0] = intrinsic_reward
            
                self.tb_logger.add_scalar('reward_model/intrinsic_reward', intrinsic_reward, self.step)
                self.tb_logger.add_scalar('reward_model/pred_gain', pred_gain, self.step)
                self.step += 1

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

        pred_prob = [target[self.index_range, indexes[i]] for i in range(batch)]

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
        print(torch.min(obs), torch.max(obs))
        assert 0 <= torch.min(obs).item() and torch.max(obs).item() <= 1

        _, x, y, _ = obs.shape
        # HWC -> CHW
        if x==y:
            obs = obs.permute(0, 3, 1, 2)

        obs = nn.functional.interpolate(obs, 42)

        obs = torch.clip(
            ((obs * self.cfg.q_level).type(torch.int64)), 0, self.cfg.q_level - 1
        )

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