from typing import Union, Dict, Optional
from easydict import EasyDict
import functools
import numpy as np
import torch
import torch.nn as nn

from ding.torch_utils.diffusion_SDE.schedule import marginal_prob_std
from ding.torch_utils.diffusion_SDE.model import ScoreNet


class QGPO(nn.Module):

    def __init__(self, cfg: EasyDict) -> None:
        super(QGPO, self).__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim

        marginal_prob_std_fn = functools.partial(marginal_prob_std, device=self.device)

        self.score_model = ScoreNet(
            cfg=cfg.score_net,
            input_dim=self.obs_dim + self.action_dim,
            output_dim=self.action_dim,
            marginal_prob_std=marginal_prob_std_fn,
        )
