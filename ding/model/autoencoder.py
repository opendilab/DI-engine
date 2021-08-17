import torch
import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self, state_dim, latent_space_dim=256, output_feature=64):
        super(AutoEncoder, self).__init__()
        self.state_dim = state_dim
        self.latent_space_dim = latent_space_dim
        self.output_feature = output_feature
        encoder_feature_dim = [96, 96, 96]
        encoder = []
        input_dim = state_dim[0]
        for i in range(len(encoder_feature_dim)):
            encoder.append(nn.Conv2d(input_dim, encoder_feature_dim[i], kernel_size=6, stride=2))
            # encoder.append(nn.BatchNorm2d(encoder_feature_dim[i]))
            encoder.append(nn.ReLU())
            input_dim = encoder_feature_dim[i]
        self.downsample = nn.Sequential(*encoder)
        before_dim, flatten_dim = self._get_flatten_size()
        self.encoder = nn.Sequential(self.downsample, nn.Flatten(), nn.Linear(flatten_dim, latent_space_dim))
        decoder = []
        decoder.append(nn.Linear(latent_space_dim, flatten_dim))
        decoder.append(nn.ReLU())
        decoder.append(nn.Unflatten(1, before_dim))
        decoder_feature_dim = [96, 96, 1]
        input_dim = encoder_feature_dim[-1]
        for i in range(len(decoder_feature_dim)):
            decoder.append(nn.ConvTranspose2d(input_dim, decoder_feature_dim[i], kernel_size=6, stride=2))
            # decoder.append(nn.BatchNorm2d(decoder_feature_dim[i]))
            decoder.append(nn.ReLU())
            input_dim = decoder_feature_dim[i]
        decoder.append(nn.Conv2d(input_dim, output_feature, kernel_size=1))
        self.decoder = nn.Sequential(*decoder)
        self.init_weight()

    def init_weight(self):
        for layer in self.encoder:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    torch.nn.init.normal_(param)
                if 'bias' in name:
                    torch.nn.init.constant_(param, 0.1)
        for layer in self.decoder:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    torch.nn.init.normal_(param)
                if 'bias' in name:
                    torch.nn.init.constant_(param, 0.1)

    def _get_flatten_size(self) -> int:
        r"""
        Overview:
            Get the encoding size after ``self.main`` to get the number of ``in-features`` to feed to ``nn.Linear``.
        Arguments:
            - x (:obj:`torch.Tensor`): Encoded Tensor after ``self.main``
        Returns:
            - outputs (:obj:`torch.Tensor`): Size int, also number of in-feature
        """
        test_data = torch.randn(1, *self.state_dim)
        with torch.no_grad():
            output = self.downsample(test_data)
        s1 = output.size()[1:]
        flatten_dim = 1
        for i in range(len(s1)):
            flatten_dim *= s1[i]
        return s1, flatten_dim

    def forward(self, input_image):
        x = input_image.view(-1, *self.state_dim)
        x = self.encoder(x)
        x = torch.sigmoid(x)
        x = self.decoder(x)
        out = torch.softmax(x + 0.003, dim=1)
        return out

    def reconstruct(self, state_emb):
        x = self.decoder(state_emb)
        out = torch.softmax(x + 0.003, dim=1)
        return out

    def generator(self, input_image):
        x = input_image.view(-1, *self.state_dim)
        x = self.encoder(x)
        x = torch.sigmoid(x)
        return x
