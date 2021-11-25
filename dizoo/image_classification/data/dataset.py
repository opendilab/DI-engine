from typing import Callable, Union
import os
import re
import math
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms


class ToNumpy:

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def find_images_and_targets(folder, types=('.png', '.jpg', '.jpeg'), class_to_idx=None, leaf_name_only=True, sort=True):
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False, followlinks=True):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)
    if class_to_idx is None:
        # building class index
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
DEFAULT_CROP_PCT = 0.875


def transforms_noaug_train(
    img_size=224,
    interpolation='bilinear',
    use_prefetcher=False,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
):
    if interpolation == 'random':
        # random interpolation not supported with no-aug
        interpolation = 'bilinear'
    tfl = [transforms.Resize(img_size, _pil_interp(interpolation)), transforms.CenterCrop(img_size)]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [transforms.ToTensor(), transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))]
    return transforms.Compose(tfl)


def transforms_imagenet_eval(
    img_size=224,
    crop_pct=None,
    interpolation='bilinear',
    use_prefetcher=False,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD
):
    crop_pct = crop_pct or DEFAULT_CROP_PCT

    if isinstance(img_size, (tuple, list)):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    tfl = [
        transforms.Resize(scale_size, _pil_interp(interpolation)),
        transforms.CenterCrop(img_size),
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [transforms.ToTensor(), transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))]

    return transforms.Compose(tfl)


class ImageNetDataset(data.Dataset):

    def __init__(self, root: str, is_training: bool, transform: Callable = None) -> None:
        self.root = root
        if transform is None:
            if is_training:
                transform = transforms_noaug_train()
            else:
                transform = transforms_imagenet_eval()
        self.transform = transform
        self.data, _ = find_images_and_targets(root)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Union[torch.Tensor, torch.Tensor]:
        img, target = self.data[index]
        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.tensor(-1, dtype=torch.long)
        return img, target
