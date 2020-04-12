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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sc2learner.torch_utils import fc_block, conv2d_block, deconv2d_block, build_activation, one_hot, LSTM, \
    ResBlock, NearestUpsample, BilinearUpsample, binary_encode, SoftArgmax
from sc2learner.torch_utils import CategoricalPdPytorch


class DelayHead(nn.Module):
    '''
        Overview: The delay head uses autoregressive_embedding to get delay_logits and delay.
        Interface: __init__, forward
    '''
    def __init__(self, cfg):
        '''
            Overview: initialize architect.
            Arguments:
                - cfg (:obj:`dict`): head architecture definition
        '''
        super(DelayHead, self).__init__()
        self.act = build_activation(cfg.activation)
        self.fc1 = fc_block(cfg.input_dim, cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc2 = fc_block(cfg.decode_dim, cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc3 = fc_block(cfg.decode_dim, 1, activation=nn.Sigmoid(), norm_type=None)  # regression
        self.embed_fc1 = fc_block(cfg.delay_encode_dim, cfg.delay_map_dim, activation=self.act, norm_type=None)
        self.embed_fc2 = fc_block(cfg.delay_map_dim, cfg.input_dim, activation=self.act, norm_type=None)

        self.delay_max_range = math.pow(2, cfg.delay_encode_dim) - 1

    def forward(self, embedding):
        '''
            Overview: This head uses autoregressive_embedding to get delay_logits. Autoregressive_embedding
                      is decoded using a 2-layer (each with size 256) linear network with ReLUs, before being
                      embedded into delay_logits that has size 128 (one for each possible requested delay in
                      game steps). Then delay is sampled from delay_logits using a multinomial, though unlike
                      all other arguments, no temperature is applied to delay_logits before sampling.
                      Delay is projected to a 1D tensor of size 1024 through a 2-layer (each with size 256)
                      linear network with ReLUs, and added to autoregressive_embedding.
            Arguments:
                - embedding (:obj:`tensor`): autoregressive_embedding
            Returns:
                - (:obj`tensor`): delay for calculation loss
                - (:obj`tensor`): delay action
                - (:obj`tensor`): autoregressive_embedding that combines information from lstm_output
                                  and all previous sampled arguments.
        '''
        x = self.fc1(embedding)
        x = self.fc2(x)
        x = self.fc3(x)
        delay = (x * self.delay_max_range).squeeze(1)

        delay_encode = binary_encode(delay, self.delay_max_range)
        embedding_delay = self.embed_fc1(delay_encode)
        embedding_delay = self.embed_fc2(embedding_delay)  # get autoregressive_embedding

        return delay, torch.round(delay).long(), embedding + embedding_delay


class QueuedHead(nn.Module):
    '''
        Overview: The queue head uses autoregressive_embedding, action_type and entity_embeddings to get
                  queued_logits and sampled queued.
        Interface: __init__, forward
    '''
    def __init__(self, cfg):
        '''
            Overview: initialize architect.
            Arguments:
                - cfg (:obj:`dict`): head architecture definition
        '''
        super(QueuedHead, self).__init__()
        self.act = build_activation(cfg.activation)
        # to get queued logits
        self.fc1 = fc_block(cfg.input_dim, cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc2 = fc_block(cfg.decode_dim, cfg.decode_dim, activation=self.act, norm_type=None)
        self.fc3 = fc_block(cfg.decode_dim, cfg.queued_dim, activation=None, norm_type=None)

        # to get autoregressive_embedding
        self.embed_fc1 = fc_block(cfg.queued_dim, cfg.queued_map_dim, activation=self.act, norm_type=None)
        self.embed_fc2 = fc_block(cfg.queued_map_dim, cfg.input_dim, activation=self.act, norm_type=None)
        self.pd = CategoricalPdPytorch

        self.queued_dim = cfg.queued_dim

    def forward(self, embedding, temperature=1.0):
        '''
            Overview: This head uses autoregressive_embedding to get queued_logits. Queued Head is similar to
                      the delay head except a temperature of 0.8 is applied to the logits before sampling, the
                      size of queued_logits is 2 (for queueing and not queueing), and the projected queued is
                      not added to autoregressive_embedding if queuing is not possible for the chosen action_type.
            Arguments:
                - embedding (:obj:`tensor`): autoregressive_embedding
                - temperature (:obj:`float`):
            Returns:
                - (:obj`tensor`): queued_logits corresponding to the probabilities of queueing and not queueing
                - (:obj`tensor`): queued that whether or no to queue this action
                - (:obj`tensor`): autoregressive_embedding that combines information from lstm_output
                                  and all previous sampled arguments.
        '''
        x = self.fc1(embedding)
        x = self.fc2(x)
        x = self.fc3(x)
        p = F.softmax(x.div(temperature), dim=1)
        handle = self.pd(p)
        if self.training:
            queued = handle.sample()
        else:
            queued = handle.mode()
        queued_one_hot = one_hot(queued, self.queued_dim)
        embedding_queued = self.embed_fc1(queued_one_hot)
        embedding_queued = self.embed_fc2(embedding_queued)  # get autoregressive_embedding

        return x, queued, embedding + embedding_queued


class SelectedUnitsHead(nn.Module):
    '''
        Overview: The selected units head uses autoregressive_embedding, action_type and entity_embeddings to get
                  units_logits and sampled units.
        Interface: __init__, forward
    '''
    def __init__(self, cfg):
        '''
            Overview: initialize architect.
            Arguments:
                - cfg (:obj:`dict`): head architecture definition
        '''
        super(SelectedUnitsHead, self).__init__()
        self.act = build_activation(cfg.activation)
        self.key_fc = fc_block(cfg.entity_embedding_dim, cfg.key_dim, activation=None, norm_type=None)
        # determines which entity types can accept action_type
        self.func_fc = fc_block(cfg.unit_type_dim, cfg.func_dim, activation=self.act, norm_type=None)
        self.fc1 = fc_block(cfg.input_dim, cfg.func_dim, activation=self.act, norm_type=None)
        self.fc2 = fc_block(cfg.func_dim, cfg.key_dim, activation=self.act, norm_type=None)
        self.embed_fc = fc_block(cfg.key_dim, cfg.input_dim, activation=None, norm_type=None)
        self.lstm = LSTM(cfg.key_dim, cfg.hidden_dim, cfg.num_layers, norm_type='LN')

        self.max_entity_num = cfg.max_entity_num
        self.key_dim = cfg.key_dim
        self.use_mask = cfg.use_mask
        self.pd = CategoricalPdPytorch

    def _get_key(self, entity_embedding):
        '''
            Overview: computes a key corresponding to each entity by feeding entity_embeddings through
                      a 1D convolution with 32 channels and kernel size 1.
            Arguments:
                - entity_embedding (:obj:`tensor`): entity embeddings
            Returns:
                - (:obj`tensor`): corresponding to ending unit selection
                - (:obj`tensor`): end_flag_index, should be the number of entity
        '''
        B, N = entity_embedding.shape[:2]
        key = self.key_fc(entity_embedding)
        end_flag = torch.zeros(B, 1, self.key_dim, device=key.device)  # zero initial state
        key = torch.cat([key, end_flag], dim=1)  # corresponding
        end_flag_index = N
        return key, end_flag_index

    def _get_init_query(self, embedding, available_unit_type_mask):
        '''
            Overview: passes autoregressive_embedding through a linear of size 256, adds func_embed, and
                      passes the combination through a ReLU and a linear of size 32.
            Arguments:
                - embedding (:obj:`tensor`): autoregressive_embedding
                - available_unit_type_mask (:obj:`tensor`): mask for available unit type
            Returns:
                - (:obj`tensor`): result
                - (:obj`tensor`): state use None as default
        '''
        func_embed = self.func_fc(available_unit_type_mask)
        x = self.fc1(embedding)
        x = self.fc2(x + func_embed)

        state = None
        return x, state

    def _query(self, key, end_flag_index, x, state, mask, temperature, output_entity_num):
        B, N = key.shape[:2]
        units = torch.zeros(B, N, device=key.device, dtype=torch.long)
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
                if self.use_mask:
                    query_result.sub_((1 - mask) * 1e9)
                p = F.softmax(query_result.div(temperature), dim=1)
                handle = self.pd(p)
                if self.training:
                    entity_num = handle.sample()
                else:
                    entity_num = handle.mode()

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
            for i in range(max(output_entity_num) + 1):
                x, state = self.lstm(x, state)
                query_result = x.permute(1, 0, 2) * key
                query_result = query_result.mean(dim=2)
                if self.use_mask:
                    query_result.sub_((1 - mask) * 1e9)
                p = F.softmax(query_result.div(temperature), dim=1)
                handle = self.pd(p)
                if self.training:
                    entity_num = handle.sample()
                else:
                    entity_num = handle.mode()
                for b in range(B):
                    if i > output_entity_num[b]:
                        continue
                    elif i < output_entity_num[b]:
                        logits[b].append(query_result)
                        units[b][entity_num[b]] = 1
                        if entity_num[b] != end_flag_index:
                            # mask[b][entity_num[b]] = 0
                            pass
                    else:
                        logits[b].append(query_result)
        embedding_selected = units.unsqueeze(2).to(key.dtype)
        embedding_selected = embedding_selected * key
        embedding_selected = embedding_selected.mean(dim=1)
        embedding_selected = self.embed_fc(embedding_selected)

        units_index = []
        for unit in units:
            index = torch.nonzero(unit).squeeze(1)
            units_index.append(index)
        return logits, units_index, embedding_selected

    def forward(
        self,
        embedding,
        available_unit_type_mask,
        available_units_mask,
        entity_embedding,
        temperature=1.0,
        output_entity_num=None
    ):
        '''
        Input:
            embedding: [batch_size, input_dim(1024)]
            available_unit_type_mask: A mask of which entity types can accept action_type, and this is a
                                      one-hot of this entity type with maximum equal to the number of unit
                                      types. [batch_size, num_unit_type]
            available_units_mask: A mask of which units can be selected, initialised to allow selecting all
                                  entities that exist (including enemy units). [batch_size, num_units]
            entity_embedding: [batch_size, num_units, entity_embedding_dim(256)]
        Output:
            logits: List(batch_size) - List(num_selected_units) - num_units
            units: [batch_size, num_units] 0-1 vector
            new_embedding: [batch_size, input_dim(1024)]
        '''
        assert (isinstance(entity_embedding, torch.Tensor))

        input, state = self._get_init_query(embedding, available_unit_type_mask)
        mask = available_units_mask
        key, end_flag_index = self._get_key(entity_embedding)
        mask = torch.cat([mask, torch.ones_like(mask[:, 0:1])], dim=1)
        logits, units, embedding_selected = self._query(
            key, end_flag_index, input, state, mask, temperature, output_entity_num
        )

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
        units = torch.zeros(B, N, device=key.device, dtype=torch.long)
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
                p = F.softmax(query_result.div(temperature), dim=1)
                handle = self.pd(p)
                if self.training:
                    entity_num = handle.sample()
                else:
                    entity_num = handle.mode()

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
            for i in range(max(output_entity_num) + 1):
                x, state = self.lstm(x, state)
                query_result = x.permute(1, 0, 2) * key
                query_result = query_result.mean(dim=2)
                query_result.sub_((1 - mask) * 1e9)
                p = F.softmax(query_result.div(temperature), dim=1)
                handle = self.pd(p)
                if self.training:
                    entity_num = handle.sample()
                else:
                    entity_num = handle.mode()
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

    def forward(
        self,
        embedding,
        available_unit_type_mask,
        available_units_mask,
        entity_embedding,
        temperature=1.0,
        output_entity_num=None
    ):
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
        assert (isinstance(entity_embedding, list) or isinstance(entity_embedding, torch.Tensor))

        input, state = self._get_init_query(embedding, available_unit_type_mask)
        mask = available_units_mask
        key, end_flag_index = self._get_key(entity_embedding)
        mask = torch.cat([mask, torch.ones_like(mask[:, 0:1])], dim=1)
        logits, units = self._query(key, end_flag_index, input, state, mask, temperature, output_entity_num)
        units_index = []
        for unit in units:
            index = torch.nonzero(unit).squeeze(1)
            units_index.append(index)

        return logits, units_index


class TargetUnitHead(nn.Module):
    '''
        Overview: The target unit head uses autoregressive_embedding to get target_unit_logits and target_unit.
        Interface: __init__, forward
    '''
    def __init__(self, cfg):
        '''
            Overview: initialize architect.
            Arguments:
                - cfg (:obj:`dict`): head architecture definition
        '''
        super(TargetUnitHead, self).__init__()
        self.act = build_activation(cfg.activation)
        self.key_fc = fc_block(cfg.entity_embedding_dim, cfg.key_dim, activation=None, norm_type=None)
        self.func_fc = fc_block(cfg.unit_type_dim, cfg.func_dim, activation=self.act, norm_type=None)
        self.fc1 = fc_block(cfg.input_dim, cfg.func_dim, activation=self.act, norm_type=None)
        self.fc2 = fc_block(cfg.func_dim, cfg.key_dim, activation=self.act, norm_type=None)
        self.use_mask = cfg.use_mask

        self.pd = CategoricalPdPytorch

    def _get_query(self, embedding, available_unit_type_mask):
        func_embed = self.func_fc(available_unit_type_mask)
        x = self.fc1(embedding)
        x = self.fc2(x + func_embed)
        return x

    def forward(self, embedding, available_unit_type_mask, available_units_mask, entity_embedding, temperature=1.0):
        '''
            Overview: First func_embed is computed the same as in the Selected Units head, and used in the
                      same way for the query (added to the output of the autoregressive_embedding passed
                      through a linear of size 256). The query is then passed through a ReLU and a linear
                      of size 32, and the query is applied to the keys which are created the same way as
                      in the Selected Units head to get target_unit_logits. target_unit is sampled from
                      target_unit_logits using a multinomial with temperature 0.8. Note that since this is
                      one of the two terminal arguments (along with Location Head, since no action has
                      both a target unit and a target location), it does not return autoregressive_embedding.
            Arguments:
                - embedding (:obj`tensor`): autoregressive_embeddingm, [batch_size, input_dim(1024)]
                - available_unit_type_mask (:obj`tensor`): [batch_size, num_unit_type]
                - available_units_mask (:obj`tensor`): [batch_size, num_units]
                - entity_embedding (:obj`tensor`): [batch_size, num_units, entity_embedding_dim(256)]
                - temperature (:obj:`float`):
            Returns:
                - (:obj`tensor`): logits, List(batch_size) - List(num_selected_units) - num_units
                - (:obj`tensor`): units, [batch_size, num_units] 0-1 vector
        '''
        assert (isinstance(entity_embedding, list) or isinstance(entity_embedding, torch.Tensor))

        mask = available_units_mask
        key = self.key_fc(entity_embedding)
        query = self._get_query(embedding, available_unit_type_mask)
        logits = query.unsqueeze(1) * key
        logits = logits.mean(dim=2)
        if self.use_mask:
            logits.sub_((1 - mask) * 1e9)

        B, N = key.shape[:2]
        units = torch.zeros(B, N, device=key.device, dtype=torch.long)
        p = F.softmax(logits.div(temperature), dim=1)
        handle = self.pd(p)
        if self.training:
            sample_num = handle.sample()
        else:
            sample_num = handle.mode()
        units.scatter_(1, sample_num.unsqueeze(1), 1)
        units_index = []
        for unit in units:
            units_index.append(torch.nonzero(unit).squeeze(1))

        return logits, units_index


class LocationHead(nn.Module):
    '''
        Overview: The location head uses autoregressive_embedding and map_skip to get target_location_logits
                  and target_location.
        Interface: __init__, forward
    '''
    def __init__(self, cfg):
        '''
            Overview: initialize architect.
            Arguments:
                - cfg (:obj:`dict`): head architecture definition
        '''
        super(LocationHead, self).__init__()
        self.act = build_activation(cfg.activation)
        self.reshape_size = cfg.reshape_size
        self.reshape_channel = cfg.reshape_channel

        self.conv1 = conv2d_block(
            cfg.map_skip_dim + cfg.reshape_channel, cfg.res_dim, 1, 1, 0, activation=self.act, norm_type=None
        )
        self.res = nn.ModuleList()
        self.res_act = nn.ModuleList()
        self.res_dim = cfg.res_dim
        for i in range(cfg.res_num):
            self.res_act.append(
                build_activation('glu')(self.res_dim, self.res_dim, cfg.map_skip_dim + cfg.reshape_channel, 'conv2d')
            )
            self.res.append(ResBlock(self.res_dim, self.res_dim, 3, 1, 1, activation=self.act, norm_type=None))

        self.upsample = nn.ModuleList()  # upsample list
        dims = [self.res_dim] + cfg.upsample_dims
        assert (cfg.upsample_type in ['deconv', 'nearest', 'bilinear'])
        for i in range(len(cfg.upsample_dims)):
            if cfg.upsample_type == 'deconv':
                self.upsample.append(deconv2d_block(dims[i], dims[i + 1], 4, 2, 1, activation=self.act, norm_type=None))
            elif cfg.upsample_type == 'nearest':
                self.upsample.append(
                    nn.Sequential(
                        NearestUpsample(2),
                        conv2d_block(dims[i], dims[i + 1], 3, 1, 1, activation=self.act, norm_type=None)
                    )
                )
            elif cfg.upsample_type == 'bilinear':
                self.upsample.append(
                    nn.Sequential(
                        BilinearUpsample(2),
                        conv2d_block(dims[i], dims[i + 1], 3, 1, 1, activation=self.act, norm_type=None)
                    )
                )

        self.use_mask = cfg.use_mask
        self.output_type = cfg.output_type
        assert (self.output_type in ['cls', 'soft_argmax'])
        if self.output_type == 'cls':
            self.pd = CategoricalPdPytorch
        else:
            self.soft_argmax = SoftArgmax()

    def forward(self, embedding, map_skip, available_location_mask, temperature=1.0):
        '''
            Overview: First autoregressive_embedding is reshaped to have the same height/width as the final skip
                      in map_skip (which was just before map information was reshaped to a 1D embedding) with 4
                      channels, and the two are concatenated together along the channel dimension, passed through
                      a ReLU, passed through a 2D convolution with 128 channels and kernel size 1, then passed
                      through another ReLU. The 3D tensor (height, width, and channels) is then passed through a
                      series of Gated ResBlocks with 128 channels, kernel size 3, and FiLM, gated on
                      autoregressive_embedding and using the elements of map_skip in order of last ResBlock skip
                      to first. Afterwards, it is upsampled 2x by each of a series of transposed 2D convolutions
                      with kernel size 4 and channel sizes 128, 64, 16, and 1 respectively (upsampled beyond the
                      128x128 input to 256x256 target location selection). Those final logits are flattened and
                      sampled (masking out invalid locations using `action_type`, such as those outside the camera
                      for build actions) with temperature 0.8 to get the actual target position.
            Arguments:
                - embedding (:obj`tensor`): autoregressive_embeddingm, [batch_size, input_dim(1024)]
                - map_skip (:obj`tensor`): tensors of the outputs of intermediate computations
                - available_location_mask (:obj`tensor`): [batch_size, num_units]
                - temperature (:obj:`float`): use 1.0 as default
            Returns:
                - (:obj`tensor`): logits
                - (:obj`tensor`): location
        '''
        reshape_embedding = embedding.reshape(-1, self.reshape_channel, *self.reshape_size)
        reshape_embedding = F.interpolate(reshape_embedding, size=map_skip[0].shape[2:], mode='bilinear')
        cat_feature = [torch.cat([reshape_embedding, map_skip[i]], dim=1) for i in range(len(map_skip))]
        x = self.act(cat_feature[-1])
        x = self.conv1(x)
        # reverse cat_feature instead of reversing resblock
        for layer, act, skip in zip(self.res, self.res_act, reversed(cat_feature)):
            x = layer(x)
            x = act(x, skip)
        for layer in self.upsample:
            x = layer(x)
        if self.use_mask:
            x -= ((1 - available_location_mask) * 1e9)
        if self.output_type == 'cls':
            logits_flatten = x.view(x.shape[0], -1)
            p = F.softmax(logits_flatten.div(temperature), dim=1)
            handle = self.pd(p)
            if self.training:
                location = handle.sample()
            else:
                location = handle.mode()

            return x, location
        elif self.output_type == 'soft_argmax':
            x = self.soft_argmax(x)
            return x, x.detach()


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
            self.output_type = 'cls'

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


def test_selected_units_head():
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


def test_target_unit_head():
    class CFG:
        def __init__(self):
            self.input_dim = 1024
            self.activation = 'relu'
            self.entity_embedding_dim = 256
            self.key_dim = 32
            self.unit_type_dim = 259
            self.func_dim = 256

    model = TargetUnitHead(CFG()).cuda()
    input = torch.randn(2, 1024).cuda()
    available_unit_type_mask = torch.ones(2, 259).cuda()
    available_units_mask = torch.ones(2, 12).cuda()
    entity_embedding = torch.randn(2, 12, 256).cuda()
    logits, units = model(input, available_unit_type_mask, available_units_mask, entity_embedding)
    print(logits.shape)
    print(units)


if __name__ == "__main__":
    test_target_unit_head()
    test_selected_units_head()
    test_location_head()
    test_delay_head()
    test_queued_head()
