import torch
import torch.nn as nn
import torch.nn.functional as F
from sc2learner.nn_utils import fc_block, conv2d_block, deconv2d_block, build_activation, one_hot, LSTM, \
    ResBlock, NearestUpsample, BilinearUpsample
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
        handle = self.pd(x.div(temperature))
        queued = handle.sample()

        queued_one_hot = one_hot(queued, self.queued_dim)
        embedding_queued = self.embed_fc1(queued_one_hot)
        embedding_queued = self.embed_fc2(embedding_queued)

        return x, queued, embedding + embedding_queued


class SelectedUnitsHead(nn.Module):
    def __init__(self, cfg):
        super(SelectedUnitsHead, self).__init__()
        self.act = build_activation(cfg.activation)
        self.key_fc = fc_block(cfg.entity_embedding_dim, cfg.key_dim, activation=None, norm_type=None)
        self.func_fc = fc_block(cfg.unit_type_dim, cfg.func_dim, activation=self.act, norm_type=None)
        self.fc1 = fc_block(cfg.input_dim, cfg.func_dim, activation=self.act, norm_type=None)
        self.fc2 = fc_block(cfg.func_dim, cfg.key_dim, activation=self.act, norm_type=None)
        self.embed_fc = fc_block(cfg.key_dim, cfg.input_dim, activation=None, norm_type=None)
        self.lstm = LSTM(cfg.key_dim, cfg.hidden_dim, cfg.num_layers, norm_type='LN')

        self.max_entity_num = cfg.max_entity_num
        self.key_dim = cfg.key_dim
        self.pd = CategoricalPdPytorch

    def _get_key(self, entity_embedding):
        B, N = entity_embedding.shape[:2]
        key = self.key_fc(entity_embedding)
        end_flag = torch.zeros(B, 1, self.key_dim, device=key.device)
        key = torch.cat([key, end_flag], dim=1)
        end_flag_index = N
        return key, end_flag_index

    def _get_init_query(self, embedding, available_unit_type_mask):
        func_embed = self.func_fc(available_unit_type_mask)
        x = self.fc1(embedding)
        x = self.fc2(x + func_embed)

        state = None
        return x, state

    def _query(self, key, end_flag_index, x, state, mask, temperature, output_entity_num):
        B, N = key.shape[:2]
        units = torch.zeros(B, N, device=key.device, dtype=torch.int)
        logits = [[] for _ in range(B)]
        x = x.unsqueeze(0)
        if output_entity_num is None:
            end_flag_trigger = [False for _ in range(B)]

            for i in range(self.max_entity_num):
                if sum(end_flag_trigger) == B:
                    break
                x, state = self.lstm(x, state)
                query_result = x.permute(1, 0, 2) * key
                query_result = query_result.mean(dim=2)
                query_result.sub_((1 - mask) * 1e9)
                handle = self.pd(query_result.div(temperature))
                entity_num = handle.sample()

                for b in range(B):
                    if end_flag_trigger[b]:
                        continue
                    else:
                        logits[b].append(query_result)
                        if entity_num[b] == end_flag_index:
                            end_flag_trigger[b] = True
                            continue
                        units[b][entity_num[b]] = 1
                        mask[b][entity_num[b]] = 0
        else:
            for i in range(max(output_entity_num)+1):
                x, state = self.lstm(x, state)
                query_result = x.permute(1, 0, 2) * key
                query_result = query_result.mean(dim=2)
                query_result.sub_((1 - mask) * 1e9)
                handle = self.pd(query_result.div(temperature))
                entity_num = handle.sample()
                for b in range(B):
                    if i > output_entity_num[b]:
                        continue
                    elif i < output_entity_num[b]:
                        logits[b].append(query_result)
                        units[b][entity_num[b]] = 1
                        if entity_num[b] != end_flag_index:
                            mask[b][entity_num[b]] = 0
                    else:
                        logits[b].append(query_result)
        embedding_selected = units.unsqueeze(2).to(key.dtype)
        embedding_selected = embedding_selected * key
        embedding_selected = embedding_selected.mean(dim=1)
        embedding_selected = self.embed_fc(embedding_selected)
        return logits, units, embedding_selected

    def forward(self, embedding, available_unit_type_mask, available_units_mask, entity_embedding,
                temperature=1.0, output_entity_num=None):
        '''
        Input:
            embedding: [batch_size, input_dim(1024)]
            available_unit_type_mask: [batch_size, num_unit_type]
            available_units_mask: [batch_size, num_units]
            entity_embedding: [batch_size, num_units, entity_embedding_dim(256)]
        Output:
            logits: List(batch_size) - List(num_selected_units) - num_units
            units: [batch_size, num_units] 0-1 vector
            new_embedding: [batch_size, input_dim(1024)]
        '''
        assert(isinstance(entity_embedding, torch.Tensor))

        input, state = self._get_init_query(embedding, available_unit_type_mask)
        mask = available_units_mask
        key, end_flag_index = self._get_key(entity_embedding)
        mask = torch.cat([mask, torch.ones_like(mask[:, 0:1])], dim=1)
        logits, units, embedding_selected = self._query(
            key, end_flag_index, input, state, mask, temperature, output_entity_num)

        return logits, units, embedding + embedding_selected


class TargetUnitsHead(nn.Module):
    def __init__(self, cfg):
        super(TargetUnitsHead, self).__init__()
        self.act = build_activation(cfg.activation)
        self.key_fc = fc_block(cfg.entity_embedding_dim, cfg.key_dim, activation=None, norm_type=None)
        self.func_fc = fc_block(cfg.unit_type_dim, cfg.func_dim, activation=self.act, norm_type=None)
        self.fc1 = fc_block(cfg.input_dim, cfg.func_dim, activation=self.act, norm_type=None)
        self.fc2 = fc_block(cfg.func_dim, cfg.key_dim, activation=self.act, norm_type=None)
        self.lstm = LSTM(cfg.key_dim, cfg.hidden_dim, cfg.num_layers, norm_type='LN')

        self.max_entity_num = cfg.max_entity_num
        self.key_dim = cfg.key_dim
        self.pd = CategoricalPdPytorch

    def _get_key(self, entity_embedding):
        B, N = entity_embedding.shape[:2]
        key = self.key_fc(entity_embedding)
        end_flag = torch.zeros(B, 1, self.key_dim, device=key.device)
        key = torch.cat([key, end_flag], dim=1)
        end_flag_index = N
        return key, end_flag_index

    def _get_init_query(self, embedding, available_unit_type_mask):
        func_embed = self.func_fc(available_unit_type_mask)
        x = self.fc1(embedding)
        x = self.fc2(x + func_embed)

        state = None
        return x, state

    def _query(self, key, end_flag_index, x, state, mask, temperature, output_entity_num):
        B, N = key.shape[:2]
        units = torch.zeros(B, N, device=key.device, dtype=torch.int)
        logits = [[] for _ in range(B)]
        x = x.unsqueeze(0)
        if output_entity_num is None:
            end_flag_trigger = [False for _ in range(B)]

            for i in range(self.max_entity_num):
                if sum(end_flag_trigger) == B:
                    break
                x, state = self.lstm(x, state)
                query_result = x.permute(1, 0, 2) * key
                query_result = query_result.mean(dim=2)
                query_result.sub_((1 - mask) * 1e9)
                handle = self.pd(query_result.div(temperature))
                entity_num = handle.sample()

                for b in range(B):
                    if end_flag_trigger[b]:
                        continue
                    else:
                        logits[b].append(query_result)
                        if entity_num[b] == end_flag_index:
                            end_flag_trigger[b] = True
                            continue
                        units[b][entity_num[b]] = 1
                        mask[b][entity_num[b]] = 0
        else:
            for i in range(max(output_entity_num)+1):
                x, state = self.lstm(x, state)
                query_result = x.permute(1, 0, 2) * key
                query_result = query_result.mean(dim=2)
                query_result.sub_((1 - mask) * 1e9)
                handle = self.pd(query_result.div(temperature))
                entity_num = handle.sample()
                for b in range(B):
                    if i > output_entity_num[b]:
                        continue
                    elif i < output_entity_num[b]:
                        logits[b].append(query_result)
                        units[b][entity_num[b]] = 1
                        if entity_num[b] != end_flag_index:
                            mask[b][entity_num[b]] = 0
                    else:
                        logits[b].append(query_result)
        return logits, units

    def forward(self, embedding, available_unit_type_mask, available_units_mask, entity_embedding,
                temperature=1.0, output_entity_num=None):
        '''
        Input:
            embedding: [batch_size, input_dim(1024)]
            available_unit_type_mask: [batch_size, num_unit_type]
            available_units_mask: [batch_size, num_units]
            entity_embedding: [batch_size, num_units, entity_embedding_dim(256)]
        Output:
            logits: List(batch_size) - List(num_selected_units) - num_units
            units: [batch_size, num_units] 0-1 vector
        '''
        assert(isinstance(entity_embedding, list) or isinstance(entity_embedding, torch.Tensor))

        input, state = self._get_init_query(embedding, available_unit_type_mask)
        mask = available_units_mask
        key, end_flag_index = self._get_key(entity_embedding)
        mask = torch.cat([mask, torch.ones_like(mask[:, 0:1])], dim=1)
        logits, units = self._query(key, end_flag_index, input, state, mask, temperature, output_entity_num)

        return logits, units


class LocationHead(nn.Module):
    def __init__(self, cfg):
        super(LocationHead, self).__init__()
        self.act = build_activation(cfg.activation)
        self.reshape_size = cfg.reshape_size
        self.reshape_channel = cfg.reshape_channel

        self.conv1 = conv2d_block(cfg.map_skip_dim+cfg.reshape_channel, cfg.res_dim, 1, 1, 0,
                                  activation=self.act, norm_type=None)
        self.res = nn.ModuleList()
        self.res_act = nn.ModuleList()
        self.res_dim = cfg.res_dim
        for i in range(cfg.res_num):
            self.res_act.append(build_activation('glu')(self.res_dim, self.res_dim,
                                                        cfg.map_skip_dim+cfg.reshape_channel, 'conv2d'))
            self.res.append(ResBlock(self.res_dim, self.res_dim, 3, 1, 1, activation=self.act, norm_type=None))

        self.upsample = nn.ModuleList()
        dims = [self.res_dim] + cfg.upsample_dims
        assert(cfg.upsample_type in ['deconv', 'nearest', 'bilinear'])
        for i in range(len(cfg.upsample_dims)):
            if cfg.upsample_type == 'deconv':
                self.upsample.append(deconv2d_block(dims[i], dims[i+1], 4, 2, 1, activation=self.act, norm_type=None))
            elif cfg.upsample_type == 'nearest':
                self.upsample.append(
                    nn.Sequential(NearestUpsample(2),
                                  conv2d_block(dims[i], dims[i+1], 3, 1, 1, activation=self.act, norm_type=None)))
            elif cfg.upsample_type == 'bilinear':
                self.upsample.append(
                    nn.Sequential(BilinearUpsample(2),
                                  conv2d_block(dims[i], dims[i+1], 3, 1, 1, activation=self.act, norm_type=None)))

        self.pd = CategoricalPdPytorch

    def forward(self, embedding, map_skip, available_location_mask, temperature=1.0):
        reshape_embedding = embedding.reshape(-1, self.reshape_channel, *self.reshape_size)
        cat_feature = [torch.cat([reshape_embedding, map_skip[i]], dim=1) for i in range(len(map_skip))]
        x = self.act(cat_feature[-1])
        x = self.conv1(x)
        for layer, act, skip in zip(self.res, self.res_act, reversed(cat_feature)):
            x = layer(x)
            x = act(x, skip)
        for layer in self.upsample:
            x = layer(x)
        x.sub_((1 - available_location_mask)*1e9)
        logits = x.view(x.shape[0], -1)
        handle = self.pd(logits.div(temperature))
        location = handle.sample()

        return logits, location


def test_location_head():
    class CFG:
        def __init__(self):
            self.activation = 'relu'
            self.upsample_type = 'deconv'
            self.upsample_dims = [128, 64, 16, 1]
            self.res_dim = 128
            self.reshape_size = (16, 16)
            self.reshape_channel = 4
            self.map_skip_dim = 128
            self.res_num = 4

    model = LocationHead(CFG()).cuda()
    embedding = torch.randn(4, 1024).cuda()
    available_location_mask = torch.ones(4, 1, 256, 256).cuda()
    map_skip = [torch.randn(4, 128, 16, 16).cuda() for _ in range(4)]
    logits, location = model(embedding, map_skip, available_location_mask)
    print(model)
    print(logits.shape)
    print(location)


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


def test_selected_unit_head():
    class CFG:
        def __init__(self):
            self.input_dim = 1024
            self.activation = 'relu'
            self.entity_embedding_dim = 256
            self.key_dim = 32
            self.unit_type_dim = 47  # temp
            self.func_dim = 256
            self.hidden_dim = 32
            self.num_layers = 1
            self.max_entity_num = 64
    model = SelectedUnitsHead(CFG()).cuda()
    print(model)
    input = torch.randn(2, 1024).cuda()
    available_unit_type_mask = torch.ones(2, 47).cuda()
    available_units_mask = torch.ones(2, 89).cuda()
    entity_embedding = torch.randn(2, 89, 256).cuda()
    logits, units, embedding = model(input, available_unit_type_mask, available_units_mask, entity_embedding)
    for b in range(2):
        print(b, len(logits[b]))
    print(logits[0][0].shape)
    print(units[0].shape, torch.nonzero(units[0]).shape)
    print(embedding.shape, input.mean(), embedding.mean())


if __name__ == "__main__":
    test_selected_unit_head()
    test_location_head()
    test_delay_head()
    test_queued_head()
