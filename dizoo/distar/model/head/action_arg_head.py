'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. Implementation for action_type_head, including basic processes.
    2. Implementation for delay_head, including basic processes.
    3. Implementation for queue_type_head, including basic processes.
    4. Implementation for selected_units_type_head, including basic processes.
    5. Implementation for target_unit_head, including basic processes.
    6. Implementation for location_head, including basic processes.
'''
from typing import Optional, List
from torch import Tensor
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.torch_utils import fc_block, conv2d_block, deconv2d_block, build_activation, ResBlock, NearestUpsample, \
    BilinearUpsample, sequence_mask, GatedConvResBlock, AttentionPool, script_lstm
from dizoo.distar.envs import MAX_ENTITY_NUM, MAX_SELECTED_UNITS_NUM


class DelayHead(nn.Module):

    def __init__(self, cfg):
        super(DelayHead, self).__init__()
        self.cfg = cfg
        self.act = build_activation(self.cfg.activation)
        self.fc1 = fc_block(self.cfg.input_dim, self.cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc2 = fc_block(self.cfg.decode_dim, self.cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc3 = fc_block(self.cfg.decode_dim, self.cfg.delay_dim, activation=None, norm_type=None)  # regression
        self.embed_fc1 = fc_block(self.cfg.delay_dim, self.cfg.delay_map_dim, activation=self.act, norm_type=None)
        self.embed_fc2 = fc_block(self.cfg.delay_map_dim, self.cfg.input_dim, activation=None, norm_type=None)

        self.delay_dim = self.cfg.delay_dim

    def forward(self, embedding, delay: Optional[torch.Tensor] = None):
        x = self.fc1(embedding)
        x = self.fc2(x)
        x = self.fc3(x)
        if delay is None:
            p = F.softmax(x, dim=1)
            delay = torch.multinomial(p, 1)[:, 0]

        delay_encode = torch.nn.functional.one_hot(delay.long(), self.delay_dim).float()
        embedding_delay = self.embed_fc1(delay_encode)
        embedding_delay = self.embed_fc2(embedding_delay)  # get autoregressive_embedding

        return x, delay, embedding + embedding_delay


class QueuedHead(nn.Module):

    def __init__(self, cfg):
        super(QueuedHead, self).__init__()
        self.cfg = cfg
        self.act = build_activation(self.cfg.activation)
        # to get queued logits
        self.fc1 = fc_block(self.cfg.input_dim, self.cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc2 = fc_block(self.cfg.decode_dim, self.cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc3 = fc_block(self.cfg.decode_dim, self.cfg.queued_dim, activation=None, norm_type=None)

        # to get autoregressive_embedding
        self.embed_fc1 = fc_block(self.cfg.queued_dim, self.cfg.queued_map_dim, activation=self.act, norm_type=None)
        self.embed_fc2 = fc_block(self.cfg.queued_map_dim, self.cfg.input_dim, activation=None, norm_type=None)

        self.queued_dim = self.cfg.queued_dim

    def forward(self, embedding, queued=None):
        x = self.fc1(embedding)
        x = self.fc2(x)
        x = self.fc3(x)
        x.div_(self.cfg.temperature)
        if queued is None:
            p = F.softmax(x, dim=1)
            queued = torch.multinomial(p, 1)[:, 0]

        queued_one_hot = torch.nn.functional.one_hot(queued.long(), self.queued_dim).float()
        embedding_queued = self.embed_fc1(queued_one_hot)
        embedding_queued = self.embed_fc2(embedding_queued)  # get autoregressive_embedding

        return x, queued, embedding + embedding_queued


class SelectedUnitsHead(nn.Module):

    def __init__(self, cfg):
        super(SelectedUnitsHead, self).__init__()
        self.cfg = cfg
        self.act = build_activation(self.cfg.activation)
        self.key_fc = fc_block(self.cfg.entity_embedding_dim, self.cfg.key_dim, activation=None, norm_type=None)
        self.query_fc1 = fc_block(self.cfg.input_dim, self.cfg.func_dim, activation=self.act)
        self.query_fc2 = fc_block(self.cfg.func_dim, self.cfg.key_dim, activation=None)
        self.embed_fc1 = fc_block(self.cfg.key_dim, self.cfg.func_dim, activation=self.act, norm_type=None)
        self.embed_fc2 = fc_block(self.cfg.func_dim, self.cfg.input_dim, activation=None, norm_type=None)

        self.max_select_num = MAX_SELECTED_UNITS_NUM
        self.max_entity_num = MAX_ENTITY_NUM
        self.key_dim = self.cfg.key_dim

        self.num_layers = self.cfg.num_layers

        self.lstm = script_lstm(self.cfg.key_dim, self.cfg.hidden_dim, self.cfg.num_layers, LN=True)
        self.end_embedding = torch.nn.Parameter(torch.FloatTensor(1, self.key_dim))
        stdv = 1. / math.sqrt(self.end_embedding.size(1))
        self.end_embedding.data.uniform_(-stdv, stdv)
        if self.cfg.entity_reduce_type == 'attention_pool':
            self.attention_pool = AttentionPool(key_dim=self.cfg.key_dim, head_num=2, output_dim=self.cfg.input_dim)
        elif self.cfg.entity_reduce_type == 'attention_pool_add_num':
            self.attention_pool = AttentionPool(
                key_dim=self.cfg.key_dim, head_num=2, output_dim=self.cfg.input_dim, max_num=MAX_SELECTED_UNITS_NUM + 1
            )
        self.extra_units = self.cfg.extra_units  # select extra units if selected units exceed max_entity_num=64

    def _get_key_mask(self, entity_embedding, entity_num):
        bs = entity_embedding.shape[0]
        padding_end = torch.zeros(1, self.end_embedding.shape[1]).repeat(bs, 1,
                                                                         1).to(entity_embedding.device)  # b, 1, c
        key = self.key_fc(entity_embedding)  # b, n, c
        key = torch.cat([key, padding_end], dim=1)
        end_embeddings = torch.ones(key.shape, dtype=key.dtype, device=key.device) * self.end_embedding.squeeze(dim=0)
        flag = torch.ones(key.shape[:2], dtype=torch.bool, device=key.device).unsqueeze(dim=2)
        flag[torch.arange(bs), entity_num] = 0
        end_embeddings = end_embeddings * ~flag
        key = key * flag
        key = key + end_embeddings
        reduce_type = self.cfg.entity_reduce_type
        if reduce_type == 'entity_num':
            key_reduce = torch.div(key, entity_num.reshape(-1, 1, 1))
            key_embeddings = self.embed_fc(key_reduce)
        elif reduce_type == 'constant':
            key_reduce = torch.div(key, 512)
            key_embeddings = self.embed_fc(key_reduce)
        elif reduce_type == 'selected_units_num' or 'attention' in reduce_type:
            key_embeddings = key
        else:
            raise NotImplementedError

        new_entity_num = entity_num + 1  # add end entity
        mask = sequence_mask(new_entity_num, max_len=entity_embedding.shape[1] + 1)
        return key, mask, key_embeddings

    def _get_pred_with_logit(self, logit):
        logit.div_(self.cfg.temperature)
        p = F.softmax(logit, dim=-1)
        units = torch.multinomial(p, 1)[:, 0]
        return units

    def _query(
        self,
        key: Tensor,
        entity_num: Tensor,
        autoregressive_embedding: Tensor,
        logits_mask: Tensor,
        key_embeddings: Tensor,
        selected_units_num: Optional[Tensor] = None,
        selected_units: Optional[Tensor] = None,
        su_mask: Tensor = None
    ):
        ae = autoregressive_embedding
        bs = ae.shape[0]
        end_flag = torch.zeros(bs, dtype=torch.bool).to(ae.device)
        results_list, logits_list = [], []
        state = [
            (torch.zeros(ae.shape[0], 32, device=ae.device), torch.zeros(ae.shape[0], 32, device=ae.device))
            for _ in range(self.num_layers)
        ]
        logits_mask[torch.arange(bs), entity_num] = torch.tensor([0], dtype=torch.bool, device=ae.device)

        result: Optional[Tensor] = None
        results: Optional[Tensor] = None
        if selected_units is not None and selected_units_num is not None:  # train
            bs = selected_units.shape[0]
            seq_len = selected_units_num.max()
            queries = []
            selected_mask = sequence_mask(selected_units_num)  # b, s
            logits_mask = logits_mask.repeat(max(seq_len, 1), 1, 1)  # b, n -> s, b, n
            logits_mask[0, torch.arange(bs), entity_num] = 0  # end flag is not available at first selection
            selected_units_one_hot = torch.zeros(*key_embeddings.shape[:2], device=ae.device).unsqueeze(dim=2)
            for i in range(max(seq_len, 1)):
                if i > 0:
                    logits_mask[i] = logits_mask[i - 1]
                    if i == 1:  # enable end flag
                        logits_mask[i, torch.arange(bs), entity_num] = 1
                    logits_mask[i, torch.arange(bs), selected_units[:, i - 1]] = 0  # mask selected units
                lstm_input = self.query_fc2(self.query_fc1(ae)).unsqueeze(0)
                lstm_output, state = self.lstm(lstm_input, state)
                queries.append(lstm_output)
                reduce_type = self.cfg.entity_reduce_type
                if reduce_type == 'selected_units_num' or 'attention' in reduce_type:
                    new_selected_units_one_hot = selected_units_one_hot.clone()  # inplace operation can not backward
                    end_flag[selected_units[:, i] == entity_num] = 1
                    new_selected_units_one_hot[torch.arange(bs)[~end_flag], selected_units[:, i][~end_flag], :] = 1
                    if reduce_type == 'selected_units_num':
                        selected_units_emebedding = (key_embeddings * new_selected_units_one_hot).sum(dim=1)
                        S = selected_units_num
                        selected_units_emebedding[S != 0] = \
                            selected_units_emebedding[S != 0] / new_selected_units_one_hot.sum(dim=1)[S != 0]
                        selected_units_emebedding = self.embed_fc2(self.embed_fc1(selected_units_emebedding))
                        ae = autoregressive_embedding + selected_units_emebedding
                    elif reduce_type == 'attention_pool':
                        ae = autoregressive_embedding + self.attention_pool(
                            key_embeddings, mask=new_selected_units_one_hot
                        )
                    elif reduce_type == 'attention_pool_add_num':
                        ae = autoregressive_embedding + self.attention_pool(
                            key_embeddings,
                            num=new_selected_units_one_hot.sum(dim=1).squeeze(dim=1),
                            mask=new_selected_units_one_hot,
                        )
                    selected_units_one_hot = new_selected_units_one_hot.clone()
                else:
                    ae = ae + key_embeddings[torch.arange(bs),
                                             selected_units[:, i]] * (i + 1 < selected_units_num).unsqueeze(1)

            queries = torch.cat(queries, dim=0).unsqueeze(dim=2)  # s, b, 1, -1
            key = key.unsqueeze(dim=0)  # 1, b, n, -1
            query_result = queries * key
            logits = query_result.sum(dim=3)  # s, b, n
            logits = logits.masked_fill(~logits_mask, -1e9)
            logits = logits.permute(1, 0, 2).contiguous()
            results = selected_units
            extra_units = None
        else:
            selected_units_num = torch.ones(bs, dtype=torch.long, device=ae.device) * self.max_select_num
            end_flag[~su_mask] = 1
            selected_units_num[~su_mask] = 0
            selected_units_one_hot = torch.zeros(*key_embeddings.shape[:2], device=ae.device).unsqueeze(dim=2)
            for i in range(self.max_select_num):
                if i > 0:
                    if i == 1:  # end flag can be selected at second selection
                        logits_mask[torch.arange(bs),
                                    entity_num] = torch.tensor([1], dtype=torch.bool, device=ae.device)
                    if result is not None:
                        logits_mask[torch.arange(bs), result.detach()] = torch.tensor(
                            [0], dtype=torch.bool, device=ae.device
                        )  # mask selected units
                lstm_input = self.query_fc2(self.query_fc1(ae)).unsqueeze(0)
                lstm_output, state = self.lstm(lstm_input, state)
                queries = lstm_output.permute(1, 0, 2)  # b, 1, c
                query_result = queries * key
                step_logits = query_result.sum(dim=2)  # b, n
                step_logits = step_logits.masked_fill(~logits_mask, -1e9)
                step_logits = step_logits.div(1)
                result = self._get_pred_with_logit(step_logits)
                selected_units_num[(result == entity_num) * ~(end_flag)] = torch.tensor(i + 1).to(result.device)
                end_flag[result == entity_num] = torch.tensor([1], dtype=torch.bool, device=ae.device)
                results_list.append(result)
                logits_list.append(step_logits)
                reduce_type = self.cfg.entity_reduce_type
                if reduce_type == 'selected_units_num' or 'attention' in reduce_type:
                    selected_units_one_hot[torch.arange(bs)[~end_flag], result[~end_flag], :] = 1
                    if reduce_type == 'selected_units_num':
                        selected_units_emebedding = (key_embeddings * selected_units_one_hot).sum(dim=1)
                        slected_num = selected_units_one_hot.sum(dim=1).squeeze(dim=1)
                        selected_units_emebedding[slected_num != 0] = selected_units_emebedding[
                            slected_num != 0] / slected_num[slected_num != 0].unsqueeze(dim=1)
                        selected_units_emebedding = self.embed_fc2(self.embed_fc1(selected_units_emebedding))
                        ae = autoregressive_embedding + selected_units_emebedding
                    elif reduce_type == 'attention_pool':
                        ae = autoregressive_embedding + self.attention_pool(key_embeddings, mask=selected_units_one_hot)
                    elif reduce_type == 'attention_pool_add_num':
                        ae = autoregressive_embedding + self.attention_pool(
                            key_embeddings,
                            num=selected_units_one_hot.sum(dim=1).squeeze(dim=1),
                            mask=selected_units_one_hot,
                        )
                else:
                    ae = ae + key_embeddings[torch.arange(bs), result] * ~end_flag.unsqueeze(dim=1)
                if end_flag.all():
                    break
            if self.extra_units:
                end_flag_logit = step_logits[torch.arange(bs), entity_num]
                extra_units = ((step_logits > end_flag_logit.unsqueeze(dim=1)) * ~end_flag.unsqueeze(dim=1)).float()
            results = torch.stack(results_list, dim=0)
            results = results.transpose(1, 0).contiguous()
            logits = torch.stack(logits_list, dim=0)
            logits = logits.transpose(1, 0).contiguous()
        return logits, results, ae, selected_units_num, extra_units

    def forward(
        self,
        embedding,
        entity_embedding,
        entity_num,
        selected_units_num: Optional[Tensor] = None,
        selected_units: Optional[Tensor] = None,
        su_mask: Optional[Tensor] = None
    ):
        key, mask, key_embeddings = self._get_key_mask(entity_embedding, entity_num)
        logits, units, embedding, selected_units_num, extra_units = self._query(
            key, entity_num, embedding, mask, key_embeddings, selected_units_num, selected_units, su_mask
        )
        return logits, units, embedding, selected_units_num, extra_units


class TargetUnitHead(nn.Module):

    def __init__(self, cfg):
        super(TargetUnitHead, self).__init__()
        self.cfg = cfg
        self.act = build_activation(self.cfg.activation)
        self.key_fc = fc_block(self.cfg.entity_embedding_dim, self.cfg.key_dim, activation=None, norm_type=None)
        self.query_fc1 = fc_block(self.cfg.input_dim, self.cfg.key_dim, activation=self.act, norm_type=None)
        self.query_fc2 = fc_block(self.cfg.key_dim, self.cfg.key_dim, activation=None, norm_type=None)
        self.key_dim = self.cfg.key_dim
        self.max_entity_num = MAX_ENTITY_NUM

    def forward(self, embedding, entity_embedding, entity_num, target_unit: Optional[torch.Tensor] = None):
        key = self.key_fc(entity_embedding)
        mask = sequence_mask(entity_num, max_len=entity_embedding.shape[1])

        query = self.query_fc2(self.query_fc1(embedding))

        logits = query.unsqueeze(1) * key
        logits = logits.sum(dim=2)  # b, n, -1
        logits.masked_fill_(~mask, value=-1e9)

        logits.div_(self.cfg.temperature)
        if target_unit is None:
            p = F.softmax(logits, dim=1)
            target_unit = torch.multinomial(p, 1)[:, 0]
        return logits, target_unit


class LocationHead(nn.Module):

    def __init__(self, cfg):
        super(LocationHead, self).__init__()
        self.cfg = cfg
        self.act = build_activation(self.cfg.activation)
        self.reshape_channel = self.cfg.reshape_channel

        self.conv1 = conv2d_block(
            self.cfg.map_skip_dim + self.cfg.reshape_channel,
            self.cfg.res_dim,
            1,
            1,
            0,
            activation=build_activation(self.cfg.activation),
            norm_type=None
        )
        self.res = nn.ModuleList()
        self.res_act = nn.ModuleList()
        self.res_dim = self.cfg.res_dim
        self.use_gate = self.cfg.gate
        self.project_embed = fc_block(
            self.cfg.input_dim,
            self.cfg.spatial_y // 8 * self.cfg.spatial_x // 8 * 4,
            activation=build_activation(self.cfg.activation)
        )

        self.res = nn.ModuleList()
        for i in range(self.cfg.res_num):
            if self.use_gate:
                self.res.append(
                    GatedConvResBlock(
                        self.res_dim,
                        self.res_dim,
                        3,
                        1,
                        1,
                        activation=build_activation(self.cfg.activation),
                        norm_type=None
                    )
                )
            else:
                self.res.append(ResBlock(self.dim, build_activation(self.cfg.activation), norm_type=None))

        self.upsample = nn.ModuleList()  # upsample list
        dims = [self.res_dim] + self.cfg.upsample_dims
        assert (self.cfg.upsample_type in ['deconv', 'nearest', 'bilinear'])
        for i in range(len(self.cfg.upsample_dims)):
            if i == len(self.cfg.upsample_dims) - 1:
                activation = None
            else:
                activation = build_activation(self.cfg.activation)
            if self.cfg.upsample_type == 'deconv':
                self.upsample.append(
                    deconv2d_block(dims[i], dims[i + 1], 4, 2, 1, activation=activation, norm_type=None)
                )
            else:
                self.upsample.append(conv2d_block(dims[i], dims[i + 1], 3, 1, 1, activation=activation, norm_type=None))

    def forward(self, embedding, map_skip: List[Tensor], location=None):
        projected_embedding = self.project_embed(embedding)
        reshape_embedding = projected_embedding.reshape(
            projected_embedding.shape[0], self.reshape_channel, self.cfg.spatial_y // 8, self.cfg.spatial_x // 8
        )
        cat_feature = torch.cat([reshape_embedding, map_skip[-1]], dim=1)

        x1 = self.act(cat_feature)
        x = self.conv1(x1)

        # reverse cat_feature instead of reversing resblock
        for i in range(self.cfg.res_num):
            x = x + map_skip[len(map_skip) - i - 1]
            if self.use_gate:
                x = self.res[i](x, x)
            else:
                x = self.res[i](x)
        for i, layer in enumerate(self.upsample):
            if self.cfg.upsample_type == 'nearest':
                x = F.interpolate(x, scale_factor=2., mode='nearest')
            elif self.cfg.upsample_type == 'bilinear':
                x = F.interpolate(x, scale_factor=2., mode='bilinear')
            x = layer(x)

        logits_flatten = x.view(x.shape[0], -1)
        logits_flatten.div_(self.cfg.temperature)
        p = F.softmax(logits_flatten, dim=1)
        if location is None:
            location = torch.multinomial(p, 1)[:, 0]
        return logits_flatten, location
