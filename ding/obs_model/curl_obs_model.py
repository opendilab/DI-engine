from typing import TYPE_CHECKING, Dict
from easydict import EasyDict
import copy
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from tensorboardX import SummaryWriter


class CURL:

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        learning_rate=1e-3,
        encoder_feature_size=50,
        target_theta=0.005,
    )

    def __init__(
            self, cfg: EasyDict, encoder: nn.Module, encoder_target: nn.Module, tb_logger: "SummaryWriter"
    ) -> None:
        """
        Overview:
            Create main components, such as paramater, optimizer, loss function and so on.
        """
        self.cfg = cfg
        self.encoder = encoder
        self.encoder_target = encoder_target
        self.tb_logger = tb_logger

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Encode original observation into more compact embedding feature.
        Arguments:
            - obs (:obj:`torch.Tensor`): Original observation returned from environment, e.g.: stacked image in Atari.
        Returns:
            - embedding (:obj:`torch.Tensor`): Embedded feature learner by CURL.
        Shapes:
            - obs: :math:`(B, C, H, W)`, where ``B = batch_size`` and ``C = frame_stack``.
            - embedding: :math:`(B, N)`, where ``N = embedding_size``
        """
        pass

    def train(self, data: Dict) -> None:
        """
        Overview:
            Compute CURL contrastive learning loss to update encoder and W.
        Arguments:
            - data (:obj:`Dict`): CURL training data, including keywords ``obs_anchor`` (:obj:`torch.Tensor`) and \
                ``obs_positive`` (:obj:`torch.Tensor`).
        """
        # prepare pair data
        obs_anchor, obs_positive = data['obs_anchor'], data['obs_positive']
        # execute CURL training

        # tb_logger record information like loss

    def state_dict(self) -> Dict:
        """
        Overview:
            Return state_dict including W and optimizer for saving checkpoint.
        """
        pass

    def load_state_dict(self, _state_dict: Dict) -> None:
        """
        Overview:
            Load state_dict including W and optimizer in order to recover.
        """
        pass
