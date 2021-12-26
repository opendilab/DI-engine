"""Note the following vae model is borrowed from https://github.com/AntixK/PyTorch-VAE"""

import torch
from torch.nn import functional as F
from torch import nn
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple

# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
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

    def __init__(self,
                 action_dim: int,
                 obs_dim: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        modules = []
        if hidden_dims is None:
            hidden_dims = [256]

        # Build Encoder
        # action
        self.action_head = nn.Sequential(
            nn.Linear(self.action_dim, hidden_dims[0]),
            nn.ReLU())
        # obs
        self.obs_head = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dims[0]),
            nn.ReLU())

        self.encoder = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[0]), nn.ReLU())
        self.mu_head = nn.Linear(hidden_dims[-1], latent_dim)
        self.var_head = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        self.condition_obs = nn.Sequential(nn.Linear(self.obs_dim, hidden_dims[0]), nn.ReLU())
        self.decoder_action = nn.Sequential(nn.Linear(latent_dim, hidden_dims[0]), nn.ReLU())
        self.decoder_common = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[0]), nn.ReLU())
        self.reconstruction_layer = nn.Sequential(nn.Linear(hidden_dims[-1], self.action_dim), nn.Tanh())  # TODO(pu): tanh

        # residual prediction
        self.prediction_head_1 = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[0]), nn.ReLU())
        self.prediction_head_2 = nn.Linear(hidden_dims[0], self.obs_dim)

        self.obs_encoding = None

    def encode(self, input) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        action_encoding = self.action_head(input['action'])
        obs_encoding = self.obs_head(input['obs'])
        # obs_encoding = self.condition_obs(input['obs'])  #  TODO(pu): using a different network

        self.obs_encoding = obs_encoding
        # input = torch.cat([obs_encoding, action_encoding], dim=-1)
        # input = obs_encoding + action_encoding  # TODO(pu): what about add, cat?
        input = obs_encoding * action_encoding

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.mu_head(result)
        log_var = self.var_head(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        action_decoding = self.decoder_action(torch.tanh(z))  # NOTE: tanh, here z is not bounded
        action_obs_decoding = action_decoding * self.obs_encoding
        action_obs_decoding_tmp = self.decoder_common(action_obs_decoding)

        reconstruction_action = self.reconstruction_layer(action_obs_decoding_tmp)
        predition_residual_tmp = self.prediction_head_1(action_obs_decoding_tmp)
        predition_residual = self.prediction_head_2(predition_residual_tmp)

        return [reconstruction_action, predition_residual]

    def decode_with_obs(self, z: Tensor, obs) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        self.obs_encoding = self.obs_head(obs)
        # TODO(pu): here z is already bounded, z is produced by td3 policy, it has been operated by tanh
        action_decoding = self.decoder_action(z)
        action_obs_decoding = action_decoding * self.obs_encoding
        action_obs_decoding_tmp = self.decoder_common(action_obs_decoding)
        reconstruction_action = self.reconstruction_layer(action_obs_decoding_tmp)
        predition_residual_tmp = self.prediction_head_1(action_obs_decoding_tmp)
        predition_residual = self.prediction_head_2(predition_residual_tmp)

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
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return {'recons_action': self.decode(z)[0], 'prediction_residual': self.decode(z)[1], 'input': input, 'mu': mu,
                'log_var': log_var, 'z': z}

    def loss_function(self,
                      args,
                      **kwargs) -> dict:
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

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
