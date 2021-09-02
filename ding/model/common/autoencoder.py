import torch
import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self, state_dim: int, latent_space_dim: int = 256, output_feature: int = 64) -> None:
        super(AutoEncoder, self).__init__()
        self.state_dim = state_dim
        self.latent_space_dim = latent_space_dim
        self.output_feature = output_feature
        encoder_feature_dim = [96, 96, 96]
        encoder = []
        input_dim = state_dim[0]
        kernel_size = [5, 5, 5]
        for i in range(len(encoder_feature_dim)):
            encoder.append(
                nn.Conv2d(input_dim, encoder_feature_dim[i], kernel_size=kernel_size[i], stride=2, padding=2)
            )
            encoder.append(nn.ReLU(inplace=True))
            input_dim = encoder_feature_dim[i]
        self.downsample = nn.Sequential(*encoder)
        before_dim, flatten_dim = self._get_flatten_size()
        self.encoder = nn.Sequential(self.downsample, nn.Flatten(), nn.Linear(flatten_dim, latent_space_dim))
        decoder = []
        decoder.append(nn.Linear(latent_space_dim, flatten_dim))
        decoder.append(nn.ReLU(inplace=True))
        decoder.append(nn.Unflatten(1, before_dim))
        decoder_feature_dim = [96, 96, 1]
        input_dim = encoder_feature_dim[-1]
        kernel_size = kernel_size[-1::-1]
        for i in range(len(decoder_feature_dim)):
            decoder.append(nn.ConvTranspose2d(input_dim, decoder_feature_dim[i], kernel_size=kernel_size[i], stride=2))
            decoder.append(nn.ReLU(inplace=True))
            input_dim = decoder_feature_dim[i]
        decoder.append(nn.Conv2d(input_dim, output_feature, kernel_size=1))
        self.decoder = nn.Sequential(*decoder)
        self.init_weight()

    def init_weight(self) -> None:
        for layer in self.encoder:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    torch.nn.init.normal_(param)
                if 'bias' in name:
                    torch.nn.init.constant_(param, 0.0)
        for layer in self.decoder:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    torch.nn.init.normal_(param)
                if 'bias' in name:
                    torch.nn.init.constant_(param, 0.0)

    def _get_flatten_size(self) -> int:
        r"""
        Overview:
            Get the encoding size after ``self.main`` to get the number of ``in-features`` to feed to ``nn.Linear``.
        Returns:
            - s1 (:obj:`torch.Tensor`): Size int, also number of in-feature
            - flatten_dim (:obj: ``torch.Tensor): Size int, dim after flatten
        """
        test_data = torch.randn(1, *self.state_dim)
        with torch.no_grad():
            output = self.downsample(test_data)
        s1 = output.size()[1:]
        flatten_dim = 1
        for i in range(len(s1)):
            flatten_dim *= s1[i]
        return s1, flatten_dim

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        r"""
        Arguments:
            - input_image (:obj:`torch.Tensor`): input images with shape (B,C,H,W)
        Returns:
            - out (:obj:`torch.Tensor`): shape (B,D,H,W)
        """
        x = input_image.view(-1, *self.state_dim)
        x = self.encoder(x)
        x = torch.sigmoid(x)
        x = self.decoder(x)
        out = torch.softmax(x + 0.003, dim=1)
        return out

    def reconstruct(self, state_emb: torch.Tensor) -> torch.Tensor:
        r"""
        Arguments:
            - state_emb (:obj:`torch.Tensor`): input images with shape (B,N0)
        Returns:
            - out (:obj:`torch.Tensor`): shape (B,D,H,W)
        """
        x = self.decoder(state_emb)
        out = torch.softmax(x + 0.003, dim=1)
        return out

    def generate(self, input_image: torch.Tensor) -> torch.Tensor:
        r"""
        Arguments:
            - input_image (:obj:`torch.Tensor`): input images with shape (B,C,H,W)
        Returns:
            - out (:obj:`torch.Tensor`): shape (B,N0)
        """
        x = input_image.view(-1, *self.state_dim)
        x = self.encoder(x)
        x = torch.sigmoid(x)
        return x
