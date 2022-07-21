"""Credit: Note the following vae model is modified from https://github.com/AntixK/PyTorch-VAE"""

import torch
from torch.nn import functional as F
from torch import nn
from abc import abstractmethod
from typing import List, Dict, Callable, Union, Any, TypeVar, Tuple, Optional
from ding.utils.type_helper import Tensor


class VanillaVAE(nn.Module):
    """
        Overview:
            Implementation of Vanilla variational autoencoder for action reconstruction.
        Interfaces:
            ``__init__``, ``encode``, ``decode``, ``decode_with_obs``, ``reparameterize``, \
                ``forward``, ``loss_function`` .
    """

    def __init__(
            self,
            action_shape: int,
            obs_shape: int,
            latent_size: int,
            hidden_dims: List = [256, 256],
            **kwargs
    ) -> None:
        super(VanillaVAE, self).__init__()
        self.action_shape = action_shape
        self.obs_shape = obs_shape
        self.latent_size = latent_size
        self.hidden_dims = hidden_dims

        # Build Encoder
        self.encode_action_head = nn.Sequential(nn.Linear(self.action_shape, hidden_dims[0]), nn.ReLU())
        self.encode_obs_head = nn.Sequential(nn.Linear(self.obs_shape, hidden_dims[0]), nn.ReLU())

        self.encode_common = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU())
        self.encode_mu_head = nn.Linear(hidden_dims[1], latent_size)
        self.encode_logvar_head = nn.Linear(hidden_dims[1], latent_size)

        # Build Decoder
        self.decode_action_head = nn.Sequential(nn.Linear(latent_size, hidden_dims[-1]), nn.ReLU())
        self.decode_common = nn.Sequential(nn.Linear(hidden_dims[-1], hidden_dims[-2]), nn.ReLU())
        # TODO(pu): tanh
        self.decode_reconst_action_head = nn.Sequential(nn.Linear(hidden_dims[-2], self.action_shape), nn.Tanh())

        # residual prediction
        self.decode_prediction_head_layer1 = nn.Sequential(nn.Linear(hidden_dims[-2], hidden_dims[-2]), nn.ReLU())
        self.decode_prediction_head_layer2 = nn.Linear(hidden_dims[-2], self.obs_shape)

        self.obs_encoding = None

    def encode(self, input: Dict[str, Tensor]) -> Dict[str, Any]:
        """
        Overview:
            Encodes the input by passing through the encoder network and returns the latent codes.
        Arguments:
            - input (:obj:`Dict`): Dict containing keywords `obs` (:obj:`torch.Tensor`) and \
                `action` (:obj:`torch.Tensor`), representing the observation and agent's action respectively.
        Returns:
            - outputs (:obj:`Dict`): Dict containing keywords ``mu`` (:obj:`torch.Tensor`), \
                ``log_var`` (:obj:`torch.Tensor`) and ``obs_encoding`` (:obj:`torch.Tensor`) \
                representing latent codes.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, O)`, where B is batch size and O is ``observation dim``.
            - action (:obj:`torch.Tensor`): :math:`(B, A)`, where B is batch size and A is ``action dim``.
            - mu (:obj:`torch.Tensor`): :math:`(B, L)`, where B is batch size and L is ``latent size``.
            - log_var (:obj:`torch.Tensor`): :math:`(B, L)`, where B is batch size and L is ``latent size``.
            - obs_encoding (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch size and H is ``hidden dim``.
        """
        action_encoding = self.encode_action_head(input['action'])
        obs_encoding = self.encode_obs_head(input['obs'])
        # obs_encoding = self.condition_obs(input['obs'])  #  TODO(pu): using a different network
        input = obs_encoding * action_encoding  # TODO(pu): what about add, cat?
        result = self.encode_common(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.encode_mu_head(result)
        log_var = self.encode_logvar_head(result)

        return {'mu': mu, 'log_var': log_var, 'obs_encoding': obs_encoding}

    def decode(self, z: Tensor, obs_encoding: Tensor) -> Dict[str, Any]:
        r"""
         Overview:
               Maps the given latent action and obs_encoding onto the original action space.
         Arguments:
             - z (:obj:`torch.Tensor`): the sampled latent action
             - obs_encoding (:obj:`torch.Tensor`): observation encoding
         Returns:
             - outputs (:obj:`Dict`): DQN forward outputs, such as q_value.
         ReturnsKeys:
             - reconstruction_action (:obj:`torch.Tensor`): reconstruction_action.
             - predition_residual (:obj:`torch.Tensor`): predition_residual.
         Shapes:
             - z (:obj:`torch.Tensor`): :math:`(B, L)`, where B is batch size and L is ``latent_size``
             - obs_encoding (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch size and H is ``hidden dim``
        """
        action_decoding = self.decode_action_head(torch.tanh(z))  # NOTE: tanh, here z is not bounded
        action_obs_decoding = action_decoding * obs_encoding
        action_obs_decoding_tmp = self.decode_common(action_obs_decoding)

        reconstruction_action = self.decode_reconst_action_head(action_obs_decoding_tmp)
        predition_residual_tmp = self.decode_prediction_head_layer1(action_obs_decoding_tmp)
        predition_residual = self.decode_prediction_head_layer2(predition_residual_tmp)
        return {'reconstruction_action': reconstruction_action, 'predition_residual': predition_residual}

    def decode_with_obs(self, z: Tensor, obs: Tensor) -> Dict[str, Any]:
        r"""
          Overview:
                Maps the given latent action and obs onto the original action space.
                Using the method self.encode_obs_head(obs) to get the obs_encoding.
          Arguments:
              - z (:obj:`torch.Tensor`): the sampled latent action
              - obs (:obj:`torch.Tensor`): observation
          Returns:
              - outputs (:obj:`Dict`): DQN forward outputs, such as q_value.
          ReturnsKeys:
              - reconstruction_action (:obj:`torch.Tensor`): the action reconstructed by VAE .
              - predition_residual (:obj:`torch.Tensor`): the observation predicted by VAE.
          Shapes:
              - z (:obj:`torch.Tensor`): :math:`(B, L)`, where B is batch size and L is ``latent_size``
              - obs (:obj:`torch.Tensor`): :math:`(B, O)`, where B is batch size and O is ``obs_shape``
        """
        obs_encoding = self.encode_obs_head(obs)
        # TODO(pu): here z is already bounded, z is produced by td3 policy, it has been operated by tanh
        action_decoding = self.decode_action_head(z)
        action_obs_decoding = action_decoding * obs_encoding
        action_obs_decoding_tmp = self.decode_common(action_obs_decoding)
        reconstruction_action = self.decode_reconst_action_head(action_obs_decoding_tmp)
        predition_residual_tmp = self.decode_prediction_head_layer1(action_obs_decoding_tmp)
        predition_residual = self.decode_prediction_head_layer2(predition_residual_tmp)

        return {'reconstruction_action': reconstruction_action, 'predition_residual': predition_residual}

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        r"""
         Overview:
              Reparameterization trick to sample from N(mu, var) from N(0,1).
         Arguments:
             - mu (:obj:`torch.Tensor`): Mean of the latent Gaussian
             - logvar (:obj:`torch.Tensor`): Standard deviation of the latent Gaussian
         Shapes:
             - mu (:obj:`torch.Tensor`): :math:`(B, L)`, where B is batch size and L is ``latnet_size``
             - logvar (:obj:`torch.Tensor`): :math:`(B, L)`, where B is batch size and L is ``latnet_size``
         """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Dict[str, Tensor], **kwargs) -> dict:
        """
        Overview:
            Encode the input, reparameterize `mu` and `log_var`, decode `obs_encoding`.
        Argumens:
            - input (:obj:`Dict`): Dict containing keywords `obs` (:obj:`torch.Tensor`) \
                and `action` (:obj:`torch.Tensor`), representing the observation \
                and agent's action respectively.
        Returns:
            - outputs (:obj:`Dict`): Dict containing keywords ``recons_action`` \
                (:obj:`torch.Tensor`), ``prediction_residual`` (:obj:`torch.Tensor`), \
                ``input`` (:obj:`torch.Tensor`), ``mu`` (:obj:`torch.Tensor`), \
                ``log_var`` (:obj:`torch.Tensor`) and ``z`` (:obj:`torch.Tensor`).
        Shapes:
            - recons_action (:obj:`torch.Tensor`): :math:`(B, A)`, where B is batch size and A is ``action dim``.
            - prediction_residual (:obj:`torch.Tensor`): :math:`(B, O)`, \
                where B is batch size and O is ``observation dim``.
            - mu (:obj:`torch.Tensor`): :math:`(B, L)`, where B is batch size and L is ``latent size``.
            - log_var (:obj:`torch.Tensor`): :math:`(B, L)`, where B is batch size and L is ``latent size``.
            - z (:obj:`torch.Tensor`): :math:`(B, L)`, where B is batch size and L is ``latent_size``
        """

        encode_output = self.encode(input)
        z = self.reparameterize(encode_output['mu'], encode_output['log_var'])
        decode_output = self.decode(z, encode_output['obs_encoding'])
        return {
            'recons_action': decode_output['reconstruction_action'],
            'prediction_residual': decode_output['predition_residual'],
            'input': input,
            'mu': encode_output['mu'],
            'log_var': encode_output['log_var'],
            'z': z
        }

    def loss_function(self, args: Dict[str, Tensor], **kwargs) -> dict:
        """
        Overview:
            Computes the VAE loss function.
            KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        Arguments:
            - args (:obj:`Dict`): Dict containing keywords `recons_action` (:obj:`torch.Tensor`) \
                and `prediction_residual` (:obj:`torch.Tensor`), `original_action` (:obj:`torch.Tensor`), \
                `mu` (:obj:`torch.Tensor`), `log_var` (:obj:`torch.Tensor`) and \
                `true_residual` (:obj:`torch.Tensor`).
            - kwargs (:obj:`Dict`): Dict containing keywords `kld_weight` (:obj:`torch.Tensor`) \
                and `predict_weight` (:obj:`torch.Tensor`).
        Returns:
            - outputs (:obj: `Dict`): Dict containing keywords `loss` \
                (`obj`:`torch.Tensor`), `reconstruction_loss` (:obj: `torch.Tensor`), \
                `kld_loss` (:obj: `torch.Tensor`) and `predict_loss` (:obj: `torch.Tensor`).
        Shapes:
            - recons_action (:obj:`torch.Tensor`): :math:`(B, A)`, where B is batch size \
                and A is ``action dim``.
            - prediction_residual (:obj:`torch.Tensor`): :math:`(B, O)`, where B is batch size \
                and O is ``observation dim``.
            - original_action (:obj:`torch.Tensor`): :math:`(B, A)`, where B is batch size and A is ``action dim``.
            - mu (:obj:`torch.Tensor`): :math:`(B, L)`, where B is batch size and L is ``latent size``.
            - log_var (:obj:`torch.Tensor`): :math:`(B, L)`, where B is batch size and L is ``latent size``.
            - true_residual (:obj:`torch.Tensor`): :math:`(B, O)`, where B is batch size and O is ``observation dim``.
        """
        recons_action = args['recons_action']
        prediction_residual = args['prediction_residual']
        original_action = args['original_action']
        mu = args['mu']
        log_var = args['log_var']
        true_residual = args['true_residual']

        kld_weight = kwargs['kld_weight']
        predict_weight = kwargs['predict_weight']

        recons_loss = F.mse_loss(recons_action, original_action)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        predict_loss = F.mse_loss(prediction_residual, true_residual)

        loss = recons_loss + kld_weight * kld_loss + predict_weight * predict_loss
        return {'loss': loss, 'reconstruction_loss': recons_loss, 'kld_loss': kld_loss, 'predict_loss': predict_loss}
