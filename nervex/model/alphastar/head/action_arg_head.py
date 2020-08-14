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
from nervex.torch_utils import fc_block, conv2d_block, deconv2d_block, build_activation, one_hot, get_lstm, \
    ResBlock, NearestUpsample, BilinearUpsample, binary_encode, SoftArgmax
from nervex.torch_utils import CategoricalPdPytorch


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
        self.fc3 = fc_block(cfg.decode_dim, cfg.delay_dim, activation=None, norm_type=None)  # regression
        self.embed_fc1 = fc_block(cfg.delay_dim, cfg.delay_map_dim, activation=self.act, norm_type=None)
        self.embed_fc2 = fc_block(cfg.delay_map_dim, cfg.input_dim, activation=None, norm_type=None)
        self.pd = CategoricalPdPytorch
        self.delay_dim = cfg.delay_dim

    def forward(self, embedding, delay=None):
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
                - delay (:obj:`tensor or None`): when SL training, the caller indicates delay value to calculate
                    embedding
            Returns:
                - (:obj`tensor`): delay for calculation loss, shape(B, delay_dim), dtype(torch.float)
                - (:obj`tensor`): delay action, shape(B, ), dtype(torch.long)
                - (:obj`tensor`): autoregressive_embedding that combines information from lstm_output
                                  and all previous sampled arguments, shape(B, input_dim)
        '''
        x = self.fc1(embedding)
        x = self.fc2(x)
        x = self.fc3(x)
        if delay is None:
            p = F.softmax(x, dim=1)
            handle = self.pd(p)
            delay = handle.sample()

        delay_encode = one_hot(delay, self.delay_dim)
        embedding_delay = self.embed_fc1(delay_encode)
        embedding_delay = self.embed_fc2(embedding_delay)  # get autoregressive_embedding

        return x, delay, embedding + embedding_delay


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
        self.embed_fc2 = fc_block(cfg.queued_map_dim, cfg.input_dim, activation=None, norm_type=None)
        self.pd = CategoricalPdPytorch

        self.queued_dim = cfg.queued_dim

    def forward(self, embedding, temperature=1.0, queued=None):
        '''
            Overview: This head uses autoregressive_embedding to get queued_logits. Queued Head is similar to
                      the delay head except a temperature of 0.8 is applied to the logits before sampling, the
                      size of queued_logits is 2 (for queueing and not queueing), and the projected queued is
                      not added to autoregressive_embedding if queuing is not possible for the chosen action_type.
            Arguments:
                - embedding (:obj:`tensor`): autoregressive_embedding
                - temperature (:obj:`float`): temperature
                - queued (obj:`tensor or None`): when SL training, the caller indicates queued to calculate embedding
            Returns:
                - (:obj`tensor`): queued_logits corresponding to the probabilities of queueing and not queueing
                - (:obj`tensor`): queued that whether or no to queue this action
                - (:obj`tensor`): autoregressive_embedding that combines information from lstm_output
                                  and all previous sampled arguments.
        '''
        x = self.fc1(embedding)
        x = self.fc2(x)
        x = self.fc3(x)
        if queued is None:
            p = F.softmax(x.div(0.8), dim=1)
            handle = self.pd(p)
            queued = handle.sample()
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
        self.fc1 = fc_block(cfg.input_dim, cfg.func_dim, activation=None, norm_type=None)
        self.fc2 = fc_block(cfg.func_dim, cfg.key_dim, activation=None, norm_type=None)
        self.embed_fc = fc_block(cfg.key_dim, cfg.input_dim, activation=None, norm_type=None)
        self.lstm = get_lstm(cfg.lstm_type, cfg.key_dim, cfg.hidden_dim, cfg.num_layers, norm_type=cfg.lstm_norm_type)

        self.max_entity_num = cfg.max_entity_num
        self.key_dim = cfg.key_dim
        self.use_mask = cfg.use_mask
        self.pd = CategoricalPdPytorch

        self.end_embedding = torch.nn.Parameter(torch.FloatTensor(1, self.key_dim))
        stdv = 1. / math.sqrt(self.end_embedding.size(1))
        self.end_embedding.data.uniform_(-stdv, stdv)

    def _get_key_mask(self, entity_embedding, entity_mask):
        '''
            Overview: computes a key corresponding to each entity by feeding entity_embeddings through
                      a 1D convolution with 32 channels and kernel size 1.
                      pad with the maximum entity number in a batch and pack into tensor
            Arguments:
                - entity_embedding (:obj:`list`): entity embeddings
                - entity_mask (:obj:`list`): entity mask
            Returns:
                - pad_key (:obj`tensor`): pad entity embeddings, which are the keys in the next match
                - pad_mask (:obj:`tensor`): pad entity mask
                - end_flag_index (:obj`tensor`): end_flag_index, should be the number of entity
        '''
        end_flag_index = [e.shape[0] for e in entity_embedding]  # 0 index
        batch_size = len(end_flag_index)
        pad_num = max(end_flag_index) + 1
        # key_fc
        key = self.key_fc(torch.cat(entity_embedding, dim=0))
        # pad key and mask, add end_flag
        pad_key = torch.zeros(batch_size, pad_num, self.key_dim, device=key.device)
        pad_mask = torch.zeros(batch_size, pad_num, device=key.device)
        start = 0
        for b in range(batch_size):
            real_num = end_flag_index[b]
            end = start + real_num
            pad_key[b, :real_num] = key[start:end]
            pad_key[b, real_num:real_num + 1] = self.end_embedding
            if self.use_mask:
                pad_mask[b, :real_num] = entity_mask[b]
            else:
                pad_mask[b, :real_num] = torch.ones_like(entity_mask[b], device=key.device)
            pad_mask[b, real_num] = 1
            start = end
        return pad_key, pad_mask, end_flag_index

    def _get_query(self, embedding, func_embed):
        '''
            Overview: passes autoregressive_embedding through a linear of size 256, adds func_embed, and
                      passes the combination through a ReLU and a linear of size 32.
            Arguments:
                - embedding (:obj:`tensor`): autoregressive_embedding
                - func_embed (:obj:`tensor`): embedding derived from available_unit_type_mask
            Returns:
                - (:obj`tensor`): result
        '''
        x = self.fc1(embedding)
        x = self.fc2(F.relu(x + func_embed))
        return x

    def _get_pred_with_logit(self, logit, temperature):
        p = F.softmax(logit.div(0.8), dim=-1)
        handle = self.pd(p)
        return handle.sample()

    def _query(self, key, end_flag_index, autoregressive_embedding, func_embed, mask, temperature, selected_units):
        B, N = key.shape[:2]
        logits = [[] for _ in range(B)]
        split_entity_num = torch.FloatTensor(end_flag_index).to(key.device).unsqueeze(1)
        if selected_units is None:
            units_index = [[] for _ in range(B)]
            end_flag_trigger = [False for _ in range(B)]
            state = None

            for i in range(self.max_entity_num):
                if sum(end_flag_trigger) == B:
                    break
                lstm_input = self._get_query(autoregressive_embedding, func_embed).unsqueeze(0)
                lstm_output, state = self.lstm(lstm_input, state)
                query_result = lstm_output.permute(1, 0, 2) * key
                query_result = query_result.sum(dim=2)
                query_result.sub_((1 - mask) * 1e9)
                if i == 0:  # mask end flag when first select
                    for b in range(B):
                        query_result[b][end_flag_index[b]] = -1e9
                entity_num = self._get_pred_with_logit(query_result, temperature)

                selected_units_step = torch.zeros(B, device=key.device, dtype=torch.long)
                for b in range(B):
                    if end_flag_trigger[b]:
                        continue
                    else:
                        logits[b].append(query_result[b])
                        # end_flag doesn't also contribute to autoregressive_embedding
                        if entity_num[b] == end_flag_index[b]:
                            end_flag_trigger[b] = True
                        else:
                            units_index[b].append(entity_num[b])
                            mask[b][entity_num[b]] = 0
                            selected_units_step[b] = entity_num[b]
                selected_mask = 1 - torch.LongTensor(end_flag_trigger).to(key.device)

                embedding_selected = one_hot(selected_units_step, N).unsqueeze(2)
                embedding_selected = embedding_selected * key
                embedding_selected = embedding_selected.sum(dim=1)
                embedding_selected = torch.div(embedding_selected, split_entity_num)  # reduce on real number
                # embedding_selected = embedding_selected.mean(dim=1)
                autoregressive_embedding = autoregressive_embedding + self.embed_fc(embedding_selected
                                                                                    ) * selected_mask.view(B, 1)

            units_index = [torch.stack(t, dim=0) for t in units_index]
        else:
            # rewrite some parts of mask which is related to selected_units label
            # TODO(nyz) output warning info about the rewrite parts
            for i, t in enumerate(selected_units):
                mask[i, t] = 1

            state = None
            output_entity_num = [t.shape[0] for t in selected_units]
            for i in range(max(output_entity_num) + 1):
                lstm_input = self._get_query(autoregressive_embedding, func_embed).unsqueeze(0)
                lstm_output, state = self.lstm(lstm_input, state)
                query_result = lstm_output.permute(1, 0, 2) * key
                query_result = query_result.sum(dim=2)
                query_result.sub_((1 - mask) * 1e9)

                # record logits and get selected_units_step
                selected_units_step = torch.zeros(B, device=key.device, dtype=torch.long)
                for b in range(B):
                    if i > output_entity_num[b]:
                        continue
                    # end_flag doesn't contribute to autoregressive_embedding
                    elif i == output_entity_num[b]:
                        logits[b].append(query_result[b])
                    else:
                        mask[b, selected_units[b][i]] = 0
                        logits[b].append(query_result[b])
                        selected_units_step[b] = selected_units[b][i]
                selected_mask = torch.LongTensor([i < num for num in output_entity_num]).to(key.device)

                embedding_selected = one_hot(selected_units_step, N).unsqueeze(2)
                embedding_selected = embedding_selected * key
                embedding_selected = embedding_selected.sum(dim=1)
                embedding_selected = torch.div(embedding_selected, split_entity_num)  # reduce on real number
                autoregressive_embedding = autoregressive_embedding + self.embed_fc(embedding_selected
                                                                                    ) * selected_mask.view(B, 1)

            units_index = selected_units

        logits = [torch.stack(t, dim=0) for t in logits]
        return logits, units_index, autoregressive_embedding

    def forward(
        self,
        embedding,
        available_unit_type_mask,
        available_units_mask,
        entity_embedding,
        temperature=1.0,
        selected_units=None
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
            selected_units: when SL training, the caller indicates selected_units to calculate embedding
        Note:
            num_units can be different among the samples in a batch, if so and batch_size > 1, unis_mask and
            entity_embedding are both list(len=batch_size) and each element is shape [1, ...]
        Output:
            logits: List(batch_size) - List(num_selected_units) - num_units
            units: [batch_size, num_units] 0-1 vector
            new_embedding: [batch_size, input_dim(1024)]
        '''

        shapes = [e.shape[0] for e in entity_embedding]
        key, mask, end_flag_index = self._get_key_mask(entity_embedding, available_units_mask)
        func_embed = self.func_fc(available_unit_type_mask)
        logits, units, embedding = self._query(
            key, end_flag_index, embedding, func_embed, mask, temperature, selected_units
        )
        logits = self._get_valid_logits(logits, shapes)

        return logits, units, embedding

    def _get_valid_logits(self, logits, shapes):
        return [t[:, :s + 1] for t, s in zip(logits, shapes)]


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
        self.fc1 = fc_block(cfg.input_dim, cfg.func_dim, activation=None, norm_type=None)
        self.fc2 = fc_block(cfg.func_dim, cfg.key_dim, activation=None, norm_type=None)
        self.use_mask = cfg.use_mask

        self.pd = CategoricalPdPytorch
        self.key_dim = cfg.key_dim

    def _get_query(self, embedding, available_unit_type_mask):
        func_embed = self.func_fc(available_unit_type_mask)
        x = self.fc1(embedding)
        x = self.fc2(F.relu(x + func_embed))
        return x

    def _get_key_mask(self, entity_embedding, entity_mask):
        '''
            Overview: computes a key corresponding to each entity by feeding entity_embeddings through
                      a 1D convolution with 32 channels and kernel size 1.
                      pad with the maximum entity number in a batch and pack into tensor
            Arguments:
                - entity_embedding (:obj:`list`): entity embeddings
                - entity_mask (:obj:`list`): entity mask
            Returns:
                - pad_key (:obj`tensor`): pad entity embeddings, which are the keys in the next match
                - pad_mask (:obj:`tensor`): pad entity mask
                - end_flag_index (:obj`tensor`): end_flag_index, should be the number of entity
        '''
        shapes = [e.shape[0] for e in entity_embedding]
        batch_size = len(entity_embedding)
        pad_num = max(shapes)
        # key_fc
        key = self.key_fc(torch.cat(entity_embedding, dim=0))
        # pad key and mask, add end_flag
        pad_key = torch.zeros(batch_size, pad_num, self.key_dim, device=key.device)
        pad_mask = torch.zeros(batch_size, pad_num, device=key.device)
        start = 0
        for b in range(batch_size):
            real_num = shapes[b]
            end = start + real_num
            pad_key[b, :real_num] = key[start:end]
            if self.use_mask:
                pad_mask[b, :real_num] = entity_mask[b]
            else:
                pad_mask[b, :real_num] = torch.ones_like(entity_mask[b], device=key.device)
            start = end
        return pad_key, pad_mask

    def forward(
        self,
        embedding,
        available_unit_type_mask,
        available_units_mask,
        entity_embedding,
        temperature=1.0,
        target_unit=None
    ):
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
                - embedding (:obj`tensor`): autoregressive_embedding, [batch_size, input_dim(1024)]
                - available_unit_type_mask (:obj`tensor`): [batch_size, num_unit_type]
                - available_units_mask (:obj`tensor`): [batch_size, num_units]
                - entity_embedding (:obj`tensor`): [batch_size, num_units, entity_embedding_dim(256)]
                - temperature (:obj:`float`): logits sample temperature
                - target_unit (:obj:`Tensor` or None): when SL training, the caller indicates target_unit
            Returns:
                - (:obj`tensor`): logits, List(batch_size) - List(num_selected_units) - num_units
                - (:obj`tensor`): target_unit, [batch_size] target_unit index
        '''
        shapes = [e.shape[0] for e in entity_embedding]
        key, mask = self._get_key_mask(entity_embedding, available_units_mask)
        query = self._get_query(embedding, available_unit_type_mask)
        logits = query.unsqueeze(1) * key
        logits = logits.sum(dim=2)
        if isinstance(target_unit, torch.Tensor):
            for i, t in enumerate(target_unit):
                if t is not None:
                    mask[i, t] = 1
        logits.sub_((1 - mask) * 1e9)

        if target_unit is None:
            p = F.softmax(logits.div(0.8), dim=1)
            target_unit = torch.multinomial(p, 1).squeeze(1)

        logits = self._get_valid_logits(logits, shapes)

        return logits, target_unit

    def _get_valid_logits(self, logits, shapes):
        logits = torch.chunk(logits, logits.shape[0], 0)
        return [t[0][:s] for t, s in zip(logits, shapes)]


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
        self.ensemble_conv = conv2d_block(dims[-1], 1, 1, 1, 0, activation=None, norm_type=None)

        self.ratio = cfg.location_expand_ratio
        self.use_mask = cfg.use_mask
        self.output_type = cfg.output_type
        assert (self.output_type in ['cls', 'soft_argmax'])
        if self.output_type == 'cls':
            self.pd = CategoricalPdPytorch
        else:
            self.soft_argmax = SoftArgmax()

    def forward(self, embedding, map_skip, available_location_mask, temperature=1.0, location=None):
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
                - embedding (:obj`tensor`): autoregressive_embedding, [batch_size, input_dim(1024)]
                - map_skip (:obj`tensor`): tensors of the outputs of intermediate computations, len=res_num, each
                    element is a torch FloatTensor with shape[batch_size, res_dim, map_y // 8, map_x // 8]
                - available_location_mask (:obj`tensor`): [batch_size, 1, map_y, map_x]
                - temperature (:obj:`float`): temperature
                - location (:obj:`Tensor`):  when SL training, the caller indicates location
            Returns:
                - (:obj`tensor`): outputs, shape[batch_size, map_y, map_x](cls), shape[batch_size, 2](soft_argmax)
                - (:obj`tensor`): location, shape[batch_size, 2]
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
        x = self.ensemble_conv(x)
        if self.use_mask:
            available_location_mask = F.interpolate(available_location_mask, size=x.shape[2:], mode='bilinear')

            available_location_mask *= (available_location_mask >= 1).float()
            x -= ((1 - available_location_mask) * 1e9)
        if self.output_type == 'cls':
            W = x.shape[3]
            logits_flatten = x.view(x.shape[0], -1)
            if location is None:
                p = F.softmax(logits_flatten.div(temperature), dim=1)
                handle = self.pd(p)
                if self.training:
                    location = handle.sample()
                else:
                    location = handle.mode()

                location = torch.stack([location // W, location % W], dim=1)
                location /= self.ratio
            x = self._map2origin_size(x)
            return x.squeeze(1), location
        elif self.output_type == 'soft_argmax':
            x = self._map2origin_size(x)
            x = self.soft_argmax(x)
            if location is None:
                location = x.detach().long()
            return x, location

    def _map2origin_size(self, x):
        if self.ratio > 1:
            r = self.ratio
            x = F.avg_pool2d(x, kernel_size=r, stride=r)
            x *= (r * r)
        return x
