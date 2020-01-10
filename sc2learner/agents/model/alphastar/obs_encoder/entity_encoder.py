import torch
import torch.nn as nn
import torch.nn.functional as F
from sc2learner.nn_utils import Transformer, fc_block, build_activation


class EntityEncoder(nn.Module):
    def __init__(self, cfg):
        super(EntityEncoder, self).__init__()
        self.act = build_activation(cfg.activation)
        self.transformer = Transformer(
            input_dim=cfg.input_dim,
            head_dim=cfg.head_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
            head_num=cfg.head_num,
            mlp_num=cfg.mlp_num,
            layer_num=cfg.layer_num,
            dropout_ratio=cfg.dropout_ratio,
            activation=self.act,
        )
        self.entity_fc = fc_block(cfg.output_dim, cfg.output_dim, activation=self.act)
        self.embed_fc = fc_block(cfg.output_dim, cfg.output_dim, activation=self.act)

    def forward(self, x):
        '''
        Input:
            x: [batch_size, entity_num, input_dim]
        Output:
            entity_embeddings: [batch_size, entity_num, output_dim]
            embedded_entity: [batch_size, output_dim]
        '''
        x = self.act(self.transformer(x))
        entity_embeddings = self.entity_fc(x)
        embedded_entity = self.embed_fc(x.mean(dim=1))  # TODO masked
        return entity_embeddings, embedded_entity


def test_entity_encoder():
    class CFG(object):
        def __init__(self):
            self.input_dim = 256  # origin 512
            self.head_dim = 128
            self.hidden_dim = 1024
            self.output_dim = 256
            self.head_num = 2
            self.mlp_num = 2
            self.layer_num = 3
            self.dropout_ratio = 0.1
            self.activation = 'relu'

    model = EntityEncoder(CFG()).cuda()
    input = torch.randn(2, 14, 256).cuda()
    entity_embeddings, embedded_entity = model(input)
    print(model)
    print(entity_embeddings.shape)
    print(embedded_entity.shape)


if __name__ == "__main__":
    test_entity_encoder()
