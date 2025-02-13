from typing import Optional, Tuple
from collections import namedtuple
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

grpo_policy_data = namedtuple('grpo_policy_data', ['logit_new', 'logit_old', 'logit_ref', 'action', 'adv', 'weight'])


def grpo_policy_error(
        data: namedtuple,
        clip_ratio: float = 0.2,
) -> Tuple[namedtuple, namedtuple]:
    """
    .. note::
        Each element in this input data is a group of response samples from the same prompt.
    """
    raise NotImplementedError
