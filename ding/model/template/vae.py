"""Note the following vae model is borrowed from https://github.com/AntixK/PyTorch-VAE"""

import torch
from torch.nn import functional as F
from torch import nn
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple, Optional
from ding.utils.type_helper import Tensor


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor, obs_encoding: Optional[Tensor]) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VanillaVAE(BaseVAE):

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
        self.condition_obs = nn.Sequential(nn.Linear(self.obs_shape, hidden_dims[-1]), nn.ReLU())
        self.decode_action_head = nn.Sequential(nn.Linear(latent_size, hidden_dims[-1]), nn.ReLU())
        self.decode_common = nn.Sequential(nn.Linear(hidden_dims[-1], hidden_dims[-2]), nn.ReLU())
        # TODO(pu): tanh
        self.decode_reconst_action_head = nn.Sequential(nn.Linear(hidden_dims[-2], self.action_shape), nn.Tanh())
        # self.decode_reconst_action_head = nn.Linear(hidden_dims[0], self.action_shape)

        # residual prediction
        self.decode_prediction_head_layer1 = nn.Sequential(nn.Linear(hidden_dims[-2], hidden_dims[-2]), nn.ReLU())
        self.decode_prediction_head_layer2 = nn.Linear(hidden_dims[-2], self.obs_shape)

        self.obs_encoding = None

    def encode(self, input) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder
        :return: (Tensor) List of latent codes
        """
        action_encoding = self.encode_action_head(input['action'])
        obs_encoding = self.encode_obs_head(input['obs'])
        # obs_encoding = self.condition_obs(input['obs'])  #  TODO(pu): using a different network

        # self.obs_encoding = obs_encoding
        # input = torch.cat([obs_encoding, action_encoding], dim=-1)
        # input = obs_encoding + action_encoding  # TODO(pu): what about add, cat?
        input = obs_encoding * action_encoding
        result = self.encode_common(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.encode_mu_head(result)
        log_var = self.encode_logvar_head(result)

        return [mu, log_var, obs_encoding]

    def decode(self, z: Tensor, obs_encoding: Tensor) -> Tensor:
        """
        Maps the given latent action codes
        onto the original action space.
          :param z: (Tensor) [B x H]
        :param obs_encoding: (Tensor) [B x H]
        :return: List(Tensor) [B x A, B x O]
        """
        action_decoding = self.decode_action_head(torch.tanh(z))  # NOTE: tanh, here z is not bounded
        # action_decoding = self.decode_action_head(z)  # NOTE: tanh, here z is not bounded
        # action_obs_decoding = action_decoding + obs_encoding  # TODO(pu): what about add, cat?
        action_obs_decoding = action_decoding * obs_encoding
        action_obs_decoding_tmp = self.decode_common(action_obs_decoding)

        reconstruction_action = self.decode_reconst_action_head(action_obs_decoding_tmp)
        predition_residual_tmp = self.decode_prediction_head_layer1(action_obs_decoding_tmp)
        predition_residual = self.decode_prediction_head_layer2(predition_residual_tmp)

        return [reconstruction_action, predition_residual]

    def decode_with_obs(self, z: Tensor, obs: Tensor) -> Tensor:
        """
        Maps the given latent action codes
        onto the original action space.
        :param z: (Tensor) [B x H]
        :param obs: (Tensor) [B x O]
        :return: List(Tensor) [B x A, B x O]
        """
        obs_encoding = self.encode_obs_head(obs)
        # TODO(pu): here z is already bounded, z is produced by td3 policy, it has been operated by tanh
        action_decoding = self.decode_action_head(z)
        # action_obs_decoding = action_decoding + obs_encoding  # TODO(pu): what about add, cat?
        action_obs_decoding = action_decoding * obs_encoding
        action_obs_decoding_tmp = self.decode_common(action_obs_decoding)
        reconstruction_action = self.decode_reconst_action_head(action_obs_decoding_tmp)
        predition_residual_tmp = self.decode_prediction_head_layer1(action_obs_decoding_tmp)
        predition_residual = self.decode_prediction_head_layer2(predition_residual_tmp)

        return [reconstruction_action, predition_residual]

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> dict:
        mu, log_var, obs_encoding = self.encode(input)
        z = self.reparameterize(mu, log_var)
        reconstruction_action, predition_residual = self.decode(z, obs_encoding)
        return {
            'recons_action': reconstruction_action,
            'prediction_residual':  predition_residual,
            'input': input,
            'mu': mu,
            'log_var': log_var,
            'z': z
        }

    def loss_function(self, args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
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


