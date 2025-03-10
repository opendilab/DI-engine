from typing import Tuple, Optional, List, Dict
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
import re

from ding.utils import REWARD_MODEL_REGISTRY
from .base_reward_model import BaseRewardModel


@REWARD_MODEL_REGISTRY.register('math_rule')
class MathRuleRewardModel(BaseRewardModel):
    config = dict(
        # (str) The type of the reward model.
        type='math_rule',
        # (str) The name of the dataset, usually the huggingface dataset name.
        dataset_name='',
        # (str) The name of the tokenizer, usually the huggingface tokenizer name.
        tokenizer_name='',
        # (float) The score of format error.
        format_error_reward=-2,
        # (float) The score of answer error.
        answer_error_reward=-1,
        # (float) The score of correct.
        correct_reward=1,
    )

    def __init__(self, config: EasyDict, device: str, logger, tb_logger: 'SummaryWriter') -> None:  # noqa
        self.cfg = config
        self.device = device
        self.logger = logger
        self.tb_logger = tb_logger

    def estimate(self, data: List[str]) -> List[Dict]:
        """
        Arguments:
            - data (:obj:`List[str]`): The list of data queries used for estimation, each query is a string of the \
                form "1 + 1 = ?"
        Returns:
            - reward (:obj:`List[Dict]`): The estimated reward.
        """
        # 1. parse the query to get question and predicted answer
        # 2. get the ground truth answer according to the question
        # 3. calculate the reward based on the predicted answer and the ground truth answer (format error -2, answer error -1, correct 1)
        pass

    # rule-based reward model does not need training, thus the following methods are empty
    def train(self):
        pass

    def collect_data(self, data: list) -> None:
        pass

    def clear_data(self) -> None:
        pass


def strip_sequence(text: str, pad_token: str, eos_token: str) -> str:
    """
    Overview:
        Remove leading and trailing sequences of padding/eos tokens from a text.

    .. note::   
        This function uses regular expressions to strip all consecutive occurrences
        of the specified padding and end-of-sequence tokens from both the beginning
        and end of the input text. Tokens in the middle of the text are preserved.

    Arguments:
        - text (str): The input text to be processed.
        - pad_token (str): The padding token to be stripped (e.g., "<PAD>").
        - eos_token (str): The end-of-sequence token to be stripped (e.g., "<EOS>").

    Returns:
        - cleaned_text (str): The cleaned text with leading/trailing padding/eos tokens removed.

    Examples:
        >>> strip_sequence("<PAD><EOS>Hello<EOS><PAD>", "<PAD>", "<EOS>")
        'Hello'

        >>> strip_sequence("Test<EOS>Middle<PAD>Keep", "<PAD>", "<EOS>")
        'Test<EOS>Middle<PAD>Keep'

        >>> strip_sequence("<EOS><EOS><PAD>Full removal<PAD><EOS>", "<PAD>", "<EOS>")
        'Full removal'

        >>> strip_sequence("No tokens here", "<PAD>", "<EOS>")
        'No tokens here'

        >>> strip_sequence("<PAD><PAD>", "<PAD>", "<EOS>")
        ''
    """
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    # Remove leading tokens
    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    # Remove trailing tokens
    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text


def normalize_text(text: str) -> str:
    """
    Overview:
        This function is designed to standardize text by:
        - Converting all text to lowercase
        - Replacing various punctuation marks and special characters with spaces
        - Removing import statements
        - Normalizing whitespace by replacing multiple spaces with a single space
        - Stripping leading and trailing whitespace
    Arguments:
        - text (str): The input text to be processed.
    Returns:
        - normalized_text (str): The normalized text.
    """
    text = re.sub("[,.:\"'\[\]\-=\+\\|!@#$%^&*();<>?/！￥…（）—\{\}：”“《》？]", " ", text.lower())
    text = re.sub("import\s[a-zA-Z\.]+(\sas\s[a-zA-Z\.]+)\n", " ", text)
    text = re.sub("\s+", " ", text)
    return text.strip()
