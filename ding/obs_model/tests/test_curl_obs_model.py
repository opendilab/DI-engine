import pytest
import numpy as np
import torch
import torch.nn as nn
from ding.obs_model import CurlObsModel


def test_curl_compute_logits():
    curl = CurlObsModel(CurlObsModel.default_config(), None)
    z_anc = torch.randn(4, 50)
    z_pos = torch.randn(4, 50)
    logits = curl.compute_logits(z_anc, z_pos)
    assert logits.shape == (4, 4)
    print('end')


def test_curl_encode():
    curl = CurlObsModel(CurlObsModel.default_config(), None)
    # x : shape : [B, C, H, W] [batch_size, 3 * frame_stack, height, width]
    x = torch.randn(64, 3 * curl.cfg.frame_stack, 84, 84)
    # embedding: :math:`(B, N)`, where ``N = embedding_size/encoder_feature_size``
    z = curl.encode(x)
    assert z.shape == (64, 50)
    print('end')


def test_curl_train():
    curl = CurlObsModel(CurlObsModel.default_config(), None)
    data = {"obs_anchor": torch.randn(64, 9, 84, 84), "obs_positive": torch.randn(64, 9, 84, 84)}
    assert curl.W.grad is None
    for p in curl.encoder.parameters():
        assert p.grad is None
    curl.train(data)
    assert curl.W.grad is not None and torch.ne(curl.W, torch.zeros(curl.W.shape)).all()  #
    for p in curl.encoder.parameters():
        assert p.grad is not None
    # print(curl.encoder.parameters())
    print('end')


def test_curl_save():
    curl = CurlObsModel(CurlObsModel.default_config(), None)
    curl.save()
    print('end')


def test_curl_load():
    curl = CurlObsModel(CurlObsModel.default_config(), None)
    curl.load()
    print('end')


def test_curl_get_augmented_data():
    curl = CurlObsModel(CurlObsModel.default_config(), None)
    img = torch.randn(64, 3 * curl.cfg.frame_stack, 100, 100)
    data = curl.get_augmented_data(img)
    assert data['obs_anchor'].shape == (64, 9, 84, 84)
