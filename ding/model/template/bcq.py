from typing import Union, Dict, Optional, List
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn

from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ..common import RegressionHead, ReparameterizationHead
from .vae import VanillaVAE


@MODEL_REGISTRY.register('bcq')
class BCQ(nn.Module):
    """
    Overview:
        Model of BCQ (Batch-Constrained deep Q-learning).
        Off-Policy Deep Reinforcement Learning without Exploration.
        https://arxiv.org/abs/1812.02900
    Interface:
        ``forward``, ``compute_actor``, ``compute_critic``, ``compute_vae``, ``compute_eval``
    Property:
        ``mode``
    """

    mode = ['compute_actor', 'compute_critic', 'compute_vae', 'compute_eval']

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType, EasyDict],
            actor_head_hidden_size: List = [400, 300],
            critic_head_hidden_size: List = [400, 300],
            activation: Optional[nn.Module] = nn.ReLU(),
            vae_hidden_dims: List = [750, 750],
            phi: float = 0.05
    ) -> None:
        """
        Overview:
            Initialize neural network, i.e. agent Q network and actor.
        Arguments:
            - obs_shape (:obj:`int`): the dimension of observation state
            - action_shape (:obj:`int`): the dimension of action shape
            - actor_hidden_size (:obj:`list`): the list of hidden size of actor
            - critic_hidden_size (:obj:'list'): the list of hidden size of critic
            - activation (:obj:`nn.Module`): Activation function in network, defaults to nn.ReLU().
            - vae_hidden_dims (:obj:`list`): the list of hidden size of vae
        """
        super(BCQ, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape = squeeze(action_shape)
        self.action_shape = action_shape
        self.input_size = obs_shape
        self.phi = phi

        critic_input_size = self.input_size + action_shape
        self.critic = nn.ModuleList()
        for _ in range(2):
            net = []
            d = critic_input_size
            for dim in critic_head_hidden_size:
                net.append(nn.Linear(d, dim))
                net.append(activation)
                d = dim
            net.append(nn.Linear(d, 1))
            self.critic.append(nn.Sequential(*net))

        net = []
        d = critic_input_size
        for dim in actor_head_hidden_size:
            net.append(nn.Linear(d, dim))
            net.append(activation)
            d = dim
        net.append(nn.Linear(d, 1))
        self.actor = nn.Sequential(*net)

        self.vae = VanillaVAE(action_shape, obs_shape, action_shape * 2, vae_hidden_dims)

    def forward(self, inputs: Dict[str, torch.Tensor], mode: str) -> Dict[str, torch.Tensor]:
        """
        Overview:
            The unique execution (forward) method of BCQ method, and one can indicate different modes to implement \
            different computation graph, including ``compute_actor`` and ``compute_critic`` in BCQ.
        Mode compute_actor:
            Arguments:
                - inputs (:obj:`Dict`): Input dict data, including obs and action tensor.
            Returns:
                - output (:obj:`Dict`): Output dict data, including action tensor.
        Mode compute_critic:
            Arguments:
                - inputs (:obj:`Dict`): Input dict data, including obs and action tensor.
            Returns:
                - output (:obj:`Dict`): Output dict data, including q_value tensor.
        Mode compute_vae:
            Arguments:
                - inputs (:obj:`Dict`): Input dict data, including obs and action tensor.
            Returns:
                - outputs (:obj:`Dict`): Dict containing keywords ``recons_action`` \
                (:obj:`torch.Tensor`), ``prediction_residual`` (:obj:`torch.Tensor`), \
                ``input`` (:obj:`torch.Tensor`), ``mu`` (:obj:`torch.Tensor`), \
                ``log_var`` (:obj:`torch.Tensor`) and ``z`` (:obj:`torch.Tensor`).
        Mode compute_eval:
            Arguments:
                - inputs (:obj:`Dict`): Input dict data, including obs and action tensor.
            Returns:
                - output (:obj:`Dict`): Output dict data, including action tensor.
        Examples:
            >>> inputs = {'obs': torch.randn(4, 32), 'action': torch.randn(4, 6)}
            >>> model = BCQ(32, 6)
            >>> outputs = model(inputs, mode='compute_actor')
            >>> outputs = model(inputs, mode='compute_critic')
            >>> outputs = model(inputs, mode='compute_vae')
            >>> outputs = model(inputs, mode='compute_eval')

        .. note::
            For specific examples, one can refer to API doc of ``compute_actor`` and ``compute_critic`` respectively.
        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_critic(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Use critic network to compute q value.
        Arguments:
            - inputs (:obj:`Dict`): Input dict data, including obs and action tensor.
        Returns:
            - outputs (:obj:`Dict`): Dict containing keywords ``q_value`` (:obj:`torch.Tensor`).
        Shapes:
            - inputs (:obj:`Dict`): :math:`(B, N, D)`, where B is batch size, N is sample number, D is input dimension.
            - outputs (:obj:`Dict`): :math:`(B, N)`.
        Examples:
            >>> inputs = {'obs': torch.randn(4, 32), 'action': torch.randn(4, 6)}
            >>> model = BCQ(32, 6)
            >>> outputs = model.compute_critic(inputs)
        """
        obs, action = inputs['obs'], inputs['action']
        if len(action.shape) == 1:  # (B, ) -> (B, 1)
            action = action.unsqueeze(1)
        x = torch.cat([obs, action], dim=-1)
        x = [m(x).squeeze() for m in self.critic]
        return {'q_value': x}

    def compute_actor(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Overview:
            Use actor network to compute action.
        Arguments:
            - inputs (:obj:`Dict`): Input dict data, including obs and action tensor.
        Returns:
            - outputs (:obj:`Dict`): Dict containing keywords ``action`` (:obj:`torch.Tensor`).
        Shapes:
            - inputs (:obj:`Dict`): :math:`(B, N, D)`, where B is batch size, N is sample number, D is input dimension.
            - outputs (:obj:`Dict`): :math:`(B, N)`.
        Examples:
            >>> inputs = {'obs': torch.randn(4, 32), 'action': torch.randn(4, 6)}
            >>> model = BCQ(32, 6)
            >>> outputs = model.compute_actor(inputs)
        """
        input = torch.cat([inputs['obs'], inputs['action']], -1)
        x = self.actor(input)
        action = self.phi * 1 * torch.tanh(x)
        action = (action + inputs['action']).clamp(-1, 1)
        return {'action': action}

    def compute_vae(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Use vae network to compute action.
        Arguments:
            - inputs (:obj:`Dict`): Input dict data, including obs and action tensor.
        Returns:
            - outputs (:obj:`Dict`): Dict containing keywords ``recons_action`` (:obj:`torch.Tensor`), \
                ``prediction_residual`` (:obj:`torch.Tensor`), ``input`` (:obj:`torch.Tensor`), \
                ``mu`` (:obj:`torch.Tensor`), ``log_var`` (:obj:`torch.Tensor`) and ``z`` (:obj:`torch.Tensor`).
        Shapes:
            - inputs (:obj:`Dict`): :math:`(B, N, D)`, where B is batch size, N is sample number, D is input dimension.
            - outputs (:obj:`Dict`): :math:`(B, N)`.
        Examples:
            >>> inputs = {'obs': torch.randn(4, 32), 'action': torch.randn(4, 6)}
            >>> model = BCQ(32, 6)
            >>> outputs = model.compute_vae(inputs)
        """
        return self.vae.forward(inputs)

    def compute_eval(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Use actor network to compute action.
        Arguments:
            - inputs (:obj:`Dict`): Input dict data, including obs and action tensor.
        Returns:
            - outputs (:obj:`Dict`): Dict containing keywords ``action`` (:obj:`torch.Tensor`).
        Shapes:
            - inputs (:obj:`Dict`): :math:`(B, N, D)`, where B is batch size, N is sample number, D is input dimension.
            - outputs (:obj:`Dict`): :math:`(B, N)`.
        Examples:
            >>> inputs = {'obs': torch.randn(4, 32), 'action': torch.randn(4, 6)}
            >>> model = BCQ(32, 6)
            >>> outputs = model.compute_eval(inputs)
        """
        obs = inputs['obs']
        obs_rep = obs.clone().unsqueeze(0).repeat_interleave(100, dim=0)
        z = torch.randn((obs_rep.shape[0], obs_rep.shape[1], self.action_shape * 2)).to(obs.device).clamp(-0.5, 0.5)
        sample_action = self.vae.decode_with_obs(z, obs_rep)['reconstruction_action']
        action = self.compute_actor({'obs': obs_rep, 'action': sample_action})['action']
        q = self.compute_critic({'obs': obs_rep, 'action': action})['q_value'][0]
        idx = q.argmax(dim=0).unsqueeze(0).unsqueeze(-1)
        idx = idx.repeat_interleave(action.shape[-1], dim=-1)
        action = action.gather(0, idx).squeeze()
        return {'action': action}
