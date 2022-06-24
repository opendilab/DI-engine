from typing import Dict, Any, List, Tuple

from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch import nn
from easydict import EasyDict

from ding.torch_utils import to_device
from ding.utils import POLICY_REGISTRY, OPTIMIZER_REGISTRY
from ding.policy.common_utils import default_preprocess_learn
from ding.policy import Policy


class StochasticOptimizer(ABC):

    @abstractmethod
    def sample(self, batch_size: int, ebm: nn.Module) -> torch.Tensor:
        """Sample counter-negatives for feeding to the InfoNCE objective."""
        raise NotImplementedError

    @abstractmethod
    def infer(self, x: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action conditioned on the current observation."""
        raise NotImplementedError


@OPTIMIZER_REGISTRY.register('vanilladf')
class VanillaDerivativeFreeOptimizer(StochasticOptimizer):
    r"""
    https://github.com/kevinzakka/ibc
    """
    pass


@OPTIMIZER_REGISTRY.register('autoregressivedf')
class AutoregressiveDerivativeFreeOptimizer(StochasticOptimizer):
    r"""
    https://github.com/conormdurkan/autoregressive-energy-machines
    """
    pass


@OPTIMIZER_REGISTRY.register('langevinmcmc')
class LangevinMCMCOptimizer(StochasticOptimizer):
    r"""
    https://github.com/google-research/ibc
    """
    pass


@POLICY_REGISTRY.register('ibc')
class IBCPolicy(Policy):

    def _init_learn(self) -> None:
        pass

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        pass

    def _state_dict_learn(self) -> Dict[str, Any]:
        pass

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        pass

    # def _init_collect(self) -> None:

    # def _forward_collect(self, data: dict) -> dict:

    # def _process_transition(self, obs: Any, policy_output: dict, timestep: namedtuple) -> dict:

    # def _get_train_sample(self, data: list) -> Union[None, List[Any]]:

    def _init_eval(self) -> None:
        pass

    def _forward_eval(self, data: dict) -> dict:
        pass

    def default_model(self) -> Tuple[str, List[str]]:
        pass

    def _monitor_vars_learn(self) -> List[str]:
        pass
