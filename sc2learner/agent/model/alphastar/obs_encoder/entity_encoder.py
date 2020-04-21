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
            x: Tensor of size [batch_size, entity_num, input_dim]
               or list of batch_size * Tensor of size [entity_num, input_dim]
               entity_num may differ for each tensor in the list
               See detailed-architecture.txt line 19-64 for more detail
               about the fields in the dim 2
        Output:
            entity_embeddings: Tensor of size [batch_size, entity_num, output_dim]
            embedded_entity: Tensor of size [batch_size, output_dim]
        '''
        if isinstance(x, list):
            num_list = [t.shape[0] for t in x]
            x = torch.cat(x, dim=0)  # cat to size [sum of entity_num, input_dim]
            x = self.act(self.transformer(x))
            entity_embeddings = self.entity_fc(x)
            x = torch.split(x, num_list, dim=0)  # split elements originated from different sample
            entity_embeddings = torch.split(entity_embeddings, num_list, dim=0)
            entity_embeddings = [t.unsqueeze(0) for t in entity_embeddings]
            embedded_entity = [t.mean(dim=0) for t in x]
            embedded_entity = torch.stack(embedded_entity)
            return entity_embeddings, embedded_entity
        else:
            x = self.act(self.transformer(x))
            entity_embeddings = self.entity_fc(x)
            embedded_entity = self.embed_fc(x.mean(dim=1))  # TODO masked
            return entity_embeddings, embedded_entity
