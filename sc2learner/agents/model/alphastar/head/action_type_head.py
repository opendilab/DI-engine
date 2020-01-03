import torch
import torch.nn as nn
import torch.nn.functional as F
from sc2learner.nn_utils import build_activation, ResFCBlock, fc_block, one_hot
from sc2learner.rl_utils import CategoricalPdPytorch


class ActionTypeHead(nn.Module):
    def __init__(self, cfg):
        super(ActionTypeHead, self).__init__()
        self.act = build_activation(cfg.activation)
        self.project = fc_block(cfg.input_dim, cfg.res_dim, activation=self.act, norm_type=cfg.norm_type)
        blocks = [ResFCBlock(cfg.res_dim, cfg.res_dim, self.act, cfg.norm_type) for _ in range(cfg.res_num)]
        self.res = nn.Sequential(*blocks)
        self.action_fc = fc_block(cfg.res_dim, cfg.action_num, activation=None, norm_type=None)

        self.action_map_fc = fc_block(cfg.action_num, cfg.action_map_dim, activation=self.act, norm_type=None)
        self.pd = CategoricalPdPytorch
        self.glu1 = build_activation('glu')(cfg.action_map_dim, cfg.gate_dim, cfg.context_dim)
        self.glu2 = build_activation('glu')(cfg.input_dim, cfg.gate_dim, cfg.context_dim)
        self.action_num = cfg.action_num

    def forward(self, lstm_output, scalar_context, temperature=1.0):
        x = self.project(lstm_output)
        x = self.res(x)
        x = self.action_fc(x)
        x.div_(temperature)
        handle = self.pd(x)
        action = handle.sample()

        action_one_hot = one_hot(action, self.action_num)
        embedding1 = self.action_map_fc(action_one_hot)
        embedding1 = self.glu1(embedding1, scalar_context)
        embedding2 = self.glu2(lstm_output, scalar_context)
        embedding = embedding1 + embedding2

        return x, action, embedding


def test_action_type_head():
    class CFG:
        def __init__(self):
            self.input_dim = 384
            self.res_dim = 256
            self.res_num = 16
            self.action_num = 314
            self.action_map_dim = 256
            self.gate_dim = 1024
            self.context_dim = 120
            self.activation = 'relu'
            self.norm_type = 'LN'

    model = ActionTypeHead(CFG()).cuda()
    lstm_output = torch.randn(4, 384).cuda()
    scalar_context = torch.randn(4, 120).cuda()
    logits, action, embedding = model(lstm_output, scalar_context)
    print(model)
    print(logits.shape)
    print(action)
    print(embedding.shape)


if __name__ == "__main__":
    test_action_type_head()
