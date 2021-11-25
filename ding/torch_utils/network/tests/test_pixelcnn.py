import torch
import numpy as np
import pytest

from ding.reward_model.count_based_model import GatedPixelCNN, GatedPixelCNNLayer, GatedActivation
from ding.torch_utils import is_differentiable

B = 1
C, H, W = 1, 41, 41
D = 8 # quant level: 8 bins


@pytest.mark.unittest
class TestPixelCNN:
    def output_check(self, model, flattened_logits, target_pixel_loss):
        loss = torch.nn.CrossEntropyLoss(reduction='none')(   # loss:[D]
            flattened_logits, target_pixel_loss.long()
        )
        loss = loss.mean()
        is_differentiable(loss, model, print_instead=True)

    def test_conv_encoder(self):
        inputs = torch.rand(B, C, H, W)
        model = GatedPixelCNN()
        print(model)
        flattened_logits, target_pixel_loss, flattened_output = model(inputs)
        self.output_check(model, flattened_logits, target_pixel_loss)
        assert flattened_logits.shape == torch.Size([B*H*W*C, D])
        assert target_pixel_loss.shape == torch.Size([B*H*W*C])
        assert flattened_output.shape == torch.Size([B*H*W*C, D])