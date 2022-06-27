"""
The following code is adapted from https://github.com/YeWR/EfficientZero/core/dataset.py
"""

import torch
import torch.nn as nn

from kornia.augmentation import RandomAffine, RandomCrop, CenterCrop, RandomResizedCrop
from kornia.filters import GaussianBlur2d


class Transforms(object):
    """
    Overview:
        Reference : Data-Efficient Reinforcement Learning with Self-Predictive Representations
        Thanks to Repo: https://github.com/mila-iqia/spr.git
    """

    def __init__(self, augmentation, shift_delta=4, image_shape=(96, 96)):
        self.augmentation = augmentation

        self.transforms = []
        for aug in self.augmentation:
            if aug == "affine":
                transformation = RandomAffine(5, (.14, .14), (.9, 1.1), (-5, 5))
            elif aug == "crop":
                transformation = RandomCrop(image_shape)
            elif aug == "rrc":
                transformation = RandomResizedCrop((100, 100), (0.8, 1))
            elif aug == "blur":
                transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
            elif aug == "shift":
                transformation = nn.Sequential(nn.ReplicationPad2d(shift_delta), RandomCrop(image_shape))
            elif aug == "intensity":
                transformation = Intensity(scale=0.05)
            elif aug == "none":
                transformation = nn.Identity()
            else:
                raise NotImplementedError()
            self.transforms.append(transformation)

    def apply_transforms(self, transforms, image):
        for transform in transforms:
            image = transform(image)
        return image

    @torch.no_grad()
    def transform(self, images):
        images = images.float() / 255. if images.dtype == torch.uint8 else images
        flat_images = images.reshape(-1, *images.shape[-3:])
        processed_images = self.apply_transforms(self.transforms, flat_images)

        processed_images = processed_images.view(*images.shape[:-3], *processed_images.shape[1:])
        return processed_images


class Intensity(nn.Module):
    """
    Overview:
        one kind of transformation to get augmentation data.
    """
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise