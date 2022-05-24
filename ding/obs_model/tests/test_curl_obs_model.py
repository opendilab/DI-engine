import pytest
import torch
import torch.nn as nn
from ding.obs_model import CurlObsModel


encoder = nn.Linear(32, 50)
encoder_target = nn.Linear(32, 50)


def test_curl_compute_logits():
    curl = CurlObsModel(CurlObsModel.default_config(), encoder, encoder_target, None)
    z_a = torch.randn(4, 50)
    z_pos = torch.randn(4, 50)
    logits = curl.compute_logits(z_a, z_pos)
    assert logits.shape == (4, 4)
    print('end')
