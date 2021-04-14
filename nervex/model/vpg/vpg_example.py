from typing import Union, List, Dict
import torch
import torch.nn as nn
from nervex.utils import Sequence
from .vpg import IVPG


class VPGNN(IVPG):
    """
    Overview:
        Basic network of a series of vanilla policy gradient algorithms (e.g.: A2C, PPO),
        which implementes the IVPG interface
    """
    def __init__(
            self,
            actor: nn.Module,
            critic: nn.Module,
            device: str = 'cpu',
    ) -> None:
        super(self, VPGNN).__init__()
        self._actor = actor
        self._critic = critic
        self._device = device

    def compute_action(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Arguments:
            - inputs (:obj:`Dict[str, torch.Tensor]`): necessary keys: ['obs']
        Returns:
            - outputs (:obj:`Dict[str, torch.Tensor]`): necessary keys: [Union['logit', 'action', 'dist']]
        """
        return self._actor(inputs['obs'])

    def compute_value(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Arguments:
            - inputs (:obj:`Dict[str, torch.Tensor]`): necessary keys: ['obs']
        Returns:
            - outputs (:obj:`Dict[str, torch.Tensor]`): necessary keys: ['value']
        """
        return self._critic(inputs['obs'])

    def compute_action_value(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Arguments:
            - inputs (:obj:`Dict[str, torch.Tensor]`): necessary keys: ['obs']
        Returns:
            - outputs (:obj:`Dict[str, torch.Tensor]`): necessary keys: ['value', Union['logit', 'action', 'dist']]
        """
        actor_output = self._actor(inputs['obs'])
        critic_output = self._critic(inputs['obs'])
        return actor_output.update(critic_output)


class VPGSharedEncoderNN(IVPG):
    """
    Overview:
        Basic network of a series of vanilla policy gradient algorithms with shared encoder(e.g.: A2C, PPO),
        which implementes the IVPG interface
    """
    def __init__(
            self,
            encoder: nn.Module,
            policy_head: nn.Module,
            value_head: nn.Module,
            device: str = 'cpu',
    ) -> None:
        super(self, VPGNN).__init__()
        self._encoder = encoder
        self._policy_head = policy_head
        self._value_head = value_head
        self._device = device

    def compute_action(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Arguments:
            - inputs (:obj:`Dict[str, torch.Tensor]`): necessary keys: ['obs']
        Returns:
            - outputs (:obj:`Dict[str, torch.Tensor]`): necessary keys: [Union['logit', 'action', 'dist']]
        """
        embedding = self._encoder(inputs['obs'])
        return self._policy_head(embedding)

    def compute_value(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Arguments:
            - inputs (:obj:`Dict[str, torch.Tensor]`): necessary keys: ['obs']
        Returns:
            - outputs (:obj:`Dict[str, torch.Tensor]`): necessary keys: ['value']
        """
        embedding = self._encoder(inputs['obs'])
        return self._value_head(embedding)

    def compute_action_value(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Arguments:
            - inputs (:obj:`Dict[str, torch.Tensor]`): necessary keys: ['obs']
        Returns:
            - outputs (:obj:`Dict[str, torch.Tensor]`): necessary keys: ['value', Union['logit', 'action', 'dist']]
        """
        embedding = self._encoder(inputs['obs'])
        policy_output = self._policy_head(embedding)
        value_output = self._value_head(embedding)
        return policy_output.update(value_output)
