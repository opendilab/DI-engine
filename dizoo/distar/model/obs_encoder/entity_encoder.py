import torch
import torch.nn as nn
from typing import Dict
from torch import Tensor

from ding.torch_utils import fc_block, build_activation, sequence_mask, Transformer
from dizoo.distar.envs import MAX_ENTITY_NUM


def get_binary_embed_mat(bit_num):
    location_embedding = []
    for n in range(2 ** bit_num):
        s = '0' * (bit_num - len(bin(n)[2:])) + bin(n)[2:]
        location_embedding.append(list(int(i) for i in s))
    return torch.tensor(location_embedding).float()


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
        self.encode_modules = nn.ModuleDict()
        self.cfg = cfg
        for k, item in self.cfg.module.items():
            if item['arc'] == 'one_hot':
                self.encode_modules[k] = nn.Embedding.from_pretrained(
                    torch.eye(item['num_embeddings']), freeze=True, padding_idx=None
                )
            if item['arc'] == 'binary':
                self.encode_modules[k] = torch.nn.Embedding.from_pretrained(
                    get_binary_embed_mat(item['num_embeddings']), freeze=True, padding_idx=None
                )
        self.act = build_activation(self.cfg.activation)
        self.transformer = Transformer(
            input_dim=self.cfg.input_dim,
            head_dim=self.cfg.head_dim,
            hidden_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.output_dim,
            head_num=self.cfg.head_num,
            mlp_num=self.cfg.mlp_num,
            layer_num=self.cfg.layer_num,
            dropout_ratio=self.cfg.dropout_ratio,
            activation=self.act,
        )
        self.entity_fc = fc_block(self.cfg.output_dim, self.cfg.output_dim, activation=self.act)
        self.embed_fc = fc_block(self.cfg.output_dim, self.cfg.output_dim, activation=self.act)
        if self.cfg.entity_reduce_type == 'attention_pool':
            from ding.torch_utils import AttentionPool
            self.attention_pool = AttentionPool(key_dim=self.cfg.output_dim, head_num=2, output_dim=self.cfg.output_dim)
        elif self.cfg.entity_reduce_type == 'attention_pool_add_num':
            from ding.torch_utils import AttentionPool
            self.attention_pool = AttentionPool(
                key_dim=self.cfg.output_dim, head_num=2, output_dim=self.cfg.output_dim, max_num=MAX_ENTITY_NUM + 1
            )

    def forward(self, x: Dict[str, Tensor], entity_num):
        entity_embedding = []
        for k, item in self.cfg.module.items():
            assert k in x.keys(), '{} not in {}'.format(k, x.keys())
            if item['arc'] == 'one_hot':
                # check data
                over_cross_data = x[k] >= item['num_embeddings']
                # if over_cross_data.any():
                #     print(k, x[k][over_cross_data])
                lower_cross_data = x[k] < 0
                if lower_cross_data.any():
                    print(k, x[k][lower_cross_data])
                    raise RuntimeError
                clipped_data = x[k].long().clamp_(max=item['num_embeddings'] - 1)
                entity_embedding.append(self.encode_modules[k](clipped_data))
            elif item['arc'] == 'binary':
                entity_embedding.append(self.encode_modules[k](x[k].long()))
            elif item['arc'] == 'unsqueeze':
                entity_embedding.append(x[k].float().unsqueeze(dim=-1))
        x = torch.cat(entity_embedding, dim=-1)
        mask = sequence_mask(entity_num, max_len=x.shape[1])
        x = self.transformer(x, mask=mask)
        entity_embeddings = self.entity_fc(self.act(x))

        if self.cfg.entity_reduce_type in ['entity_num', 'selected_units_num']:
            x_mask = x * mask.unsqueeze(dim=2)
            embedded_entity = x_mask.sum(dim=1) / entity_num.unsqueeze(dim=-1)
        elif self.cfg.entity_reduce_type == 'constant':
            x_mask = x * mask.unsqueeze(dim=2)
            embedded_entity = x_mask.sum(dim=1) / 512
        elif self.cfg.entity_reduce_type == 'attention_pool':
            embedded_entity = self.attention_pool(x, mask=mask.unsqueeze(dim=2))
        elif self.cfg.entity_reduce_type == 'attention_pool_add_num':
            embedded_entity = self.attention_pool(x, num=entity_num, mask=mask.unsqueeze(dim=2))
        else:
            raise NotImplementedError
        embedded_entity = self.embed_fc(embedded_entity)
        return entity_embeddings, embedded_entity, mask
