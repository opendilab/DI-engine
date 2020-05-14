from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from sc2learner.torch_utils import Transformer, fc_block, build_activation


class EntityEncoder(nn.Module):
    r'''
    B=batch size EN=any number of entities ID=input_dim OS=output_size=256
     (B*EN*ID)  (EN'*OS)          (EN'*OS)          (EN'*OS)           (B*EN*OS)
    x -> combine -> Transformer ->  act ->  entity_fc  -> split ->   entity_embeddings
          batch                         |      (B*EN*OS)   (B*OS)        (B*OS)
                                        \->  split ->  mean -> embed_fc -> embedded_entity
    '''
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
            x: list(tuple) of batch_size * Tensor of size [entity_num, input_dim]
               entity_num may differ for each tensor in the list
               See detailed-architecture.txt line 19-64 for more detail
               about the fields in the dim 2
        Output:
            entity_embeddings: tuple(len=batch_size)->element: torch.Tensor, shape(entity_num_b, output_dim)
            embedded_entity: Tensor of size [batch_size, output_dim]
        '''
        x, valid_num = self.transformer(x, tensor_output=True)
        x = self.act(x)
        entity_embeddings = self.entity_fc(x)
        entity_embeddings = [e[:v] for e, v in zip(entity_embeddings, valid_num)]
        embedded_entity = [t[:v].mean(dim=0) for t, v in zip(x, valid_num)]
        embedded_entity = torch.stack(embedded_entity)
        embedded_entity = self.embed_fc(embedded_entity)
        return entity_embeddings, embedded_entity
