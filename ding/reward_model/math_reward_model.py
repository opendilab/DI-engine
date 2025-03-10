from typing import Tuple, Optional, List, Dict
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
import re

from ding.utils import REWARD_MODEL_REGISTRY
from .base_reward_model import BaseRewardModel


@REWARD_MODEL_REGISTRY.register('math')
class MathRewardModel(BaseRewardModel):
    config = dict(
        # (str) The type of the reward model.
        type='math',
        # (str) The name of the tokenizer, usually the huggingface tokenizer name.
        tokenizer_name='Qwen/Qwen2.5-Math-PRM-7B',
    )

    def __init__(self, config: EasyDict, device: str, logger, tb_logger: 'SummaryWriter') -> None:  # noqa
        self.cfg = config
        self.device = device
        self.logger = logger
        self.tb_logger = tb_logger

    def estimate(self, data: List[str]) -> List[Dict]:
        """
        Arguments:
            - data (:obj:`List[str]`): The list of data queries used for estimation, each query is a string.
              of the \
                form "1 + 1 = ?"
        Returns:
            - reward (:obj:`List[Dict]`): The estimated reward.
        """
        pass

    # rule-based reward model does not need training, thus the following methods are empty
    def train(self):
        pass

    def collect_data(self, data: list) -> None:
        pass

    def clear_data(self) -> None:
        pass