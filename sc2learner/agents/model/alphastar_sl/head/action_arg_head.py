import torch
import torch.nn as nn
import torch.nn.functional as F
from sc2learner.nn_utils import fc_block, build_activation, one_hot
from sc2learner.rl_utils import CategoricalPdPytorch


class DelayHead(nn.Module):
    def __init__(self, cfg):
        super(DelayHead, self).__init__()
        self.act = build_activation(cfg.activation)
        self.fc1 = fc_block(cfg.input_dim, cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc2 = fc_block(cfg.decode_dim, cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc3 = fc_block(cfg.decode_dim, cfg.delay_dim, activation=None, norm_type=None)
        self.embed_fc1 = fc_block(cfg.delay_dim, cfg.delay_map_dim, activation=self.act, norm_type=None)
        self.embed_fc2 = fc_block(cfg.delay_map_dim, cfg.input_dim, activation=self.act, norm_type=None)
        self.pd = CategoricalPdPytorch

        self.delay_dim = cfg.delay_dim

    def forward(self, embedding):
        x = self.fc1(embedding)
        x = self.fc2(x)
        x = self.fc3(x)
        handle = self.pd(x)
        delay = handle.sample()

        delay_one_hot = one_hot(delay, self.delay_dim)
        embedding_delay = self.embed_fc1(delay_one_hot)
        embedding_delay = self.embed_fc2(embedding_delay)

        return x, delay, embedding + embedding_delay


class QueuedHead(nn.Module):
    def __init__(self, cfg):
        super(QueuedHead, self).__init__()
        self.act = build_activation(cfg.activation)
        self.fc1 = fc_block(cfg.input_dim, cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc2 = fc_block(cfg.decode_dim, cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc3 = fc_block(cfg.decode_dim, cfg.queued_dim, activation=None, norm_type=None)
        self.embed_fc1 = fc_block(cfg.queued_dim, cfg.queued_map_dim, activation=self.act, norm_type=None)
        self.embed_fc2 = fc_block(cfg.queued_map_dim, cfg.input_dim, activation=self.act, norm_type=None)
        self.pd = CategoricalPdPytorch

        self.queued_dim = cfg.queued_dim

    def forward(self, embedding, temperature=1.0):
        x = self.fc1(embedding)
        x = self.fc2(x)
        x = self.fc3(x)
        x.div_(temperature)
        handle = self.pd(x)
        queued = handle.sample()

        queued_one_hot = one_hot(queued, self.queued_dim)
        embedding_queued = self.embed_fc1(queued_one_hot)
        embedding_queued = self.embed_fc2(embedding_queued)

        return x, queued, embedding + embedding_queued


def test_delay_head():
    class CFG:
        def __init__(self):
            self.input_dim = 1024
            self.decode_dim = 256
            self.delay_dim = 128
            self.delay_map_dim = 256
            self.activation = 'relu'

    model = DelayHead(CFG()).cuda()
    input = torch.randn(4, 1024).cuda()
    logits, delay, embedding = model(input)
    print(model)
    print(logits.shape)
    print(delay)
    print(embedding.shape)
    print(input.mean(), embedding.mean())


def test_queued_head():
    class CFG:
        def __init__(self):
            self.input_dim = 1024
            self.decode_dim = 256
            self.queued_dim = 2
            self.queued_map_dim = 256
            self.activation = 'relu'

    model = QueuedHead(CFG()).cuda()
    input = torch.randn(4, 1024).cuda()
    logits, queued, embedding = model(input)
    print(model)
    print(logits.shape)
    print(queued)
    print(embedding.shape)
    print(input.mean(), embedding.mean())


if __name__ == "__main__":
    test_delay_head()
    test_queued_head()
