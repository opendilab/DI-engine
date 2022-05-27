import pytest
import torch
import torch.nn as nn
from ding.obs_model import *
# from ding.obs_model import CurlObsModel
# from ding.obs_model import Encoder
from torch.autograd import Variable

# encoder = nn.Linear(32, 50)
# encoder_target = nn.Linear(32, 50)

def test_curl_compute_logits():
    curl = CurlObsModel(CurlObsModel.default_config(), None)
    z_anc = torch.randn(4, 50)
    z_pos = torch.randn(4, 50)
    logits = curl.compute_logits(z_anc, z_pos)
    assert logits.shape == (4, 4)
    print('end')

def test_curl_encode():
    curl = CurlObsModel(CurlObsModel.default_config(), None)
    # x : shape : [B, C, H, W] [batch_size, frame_stack, height, width]
    x = torch.FloatTensor(64, 3 * curl.cfg.frame_stack, 84, 84)
    x = Variable(x)
    # embedding: :math:`(B, N)`, where ``N = embedding_size/encoder_feature_size``
    z = curl.encode(x)
    assert  z.shape == (64, 50)
    print('end')


def test_curl_train():
    curl = CurlObsModel(CurlObsModel.default_config(), None)
    data = {"obs_anchor": torch.FloatTensor(64, 9, 84, 84) ,
            "obs_positive": torch.FloatTensor(64, 9, 84, 84)}
    curl.train(data)
    #怎么assert？
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
