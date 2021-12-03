class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


#######

def init_mems(self):
    if self.mem_len > 0:
        mems = []
        param = next(self.parameters())
        for i in range(self.n_layer+1):
            empty = torch.empty(0, dtype=param.dtype, device=param.device)
            mems.append(empty)

        return mems
    else:
        return None

def _update_mems(self, hids, mems, qlen, mlen):
    # does not deal with None
    if mems is None: return None

    # mems is not None
    assert len(hids) == len(mems), 'len(hids) != len(mems)'

    # There are `mlen + qlen` steps that can be cached into mems
    # For the next step, the last `ext_len` of the `qlen` tokens
    # will be used as the extended context. Hence, we only cache
    # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
    # to `mlen + qlen - self.ext_len`.
    with torch.no_grad():
        new_mems = []
        end_idx = mlen + max(0, qlen - 0 - self.ext_len)
        beg_idx = max(0, end_idx - self.mem_len)
        for i in range(len(hids)):

            cat = torch.cat([mems[i], hids[i]], dim=0)
            new_mems.append(cat[beg_idx:end_idx].detach())

    return new_mems

####################

qlen, bsz = dec_inp.size()

word_emb = self.word_emb(dec_inp)

mlen = mems[0].size(0) if mems is not None else 0
klen = mlen + qlen
if self.same_length:
    all_ones = word_emb.new_ones(qlen, klen)
    mask_len = klen - self.mem_len
    if mask_len > 0:
        mask_shift_len = qlen - mask_len
    else:
        mask_shift_len = qlen
    dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                     + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None] # -1
else:
    dec_attn_mask = torch.triu(
        word_emb.new_ones(qlen, klen), diagonal=1+mlen).byte()[:,:,None]

hids = []
if self.attn_type == 0: # default
    pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device,
                           dtype=word_emb.dtype)
    if self.clamp_len > 0:
        pos_seq.clamp_(max=self.clamp_len)
    pos_emb = self.pos_emb(pos_seq)

    core_out = self.drop(word_emb)
    pos_emb = self.drop(pos_emb)

    hids.append(core_out)
    for i, layer in enumerate(self.layers):
        mems_i = None if mems is None else mems[i]
        core_out = layer(core_out, pos_emb, self.r_w_bias,
                         self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
        hids.append(core_out)
elif self.attn_type == 1: # learnable
    core_out = self.drop(word_emb)
    hids.append(core_out)
    for i, layer in enumerate(self.layers):
        if self.clamp_len > 0:
            r_emb = self.r_emb[i][-self.clamp_len :]
            r_bias = self.r_bias[i][-self.clamp_len :]
        else:
            r_emb, r_bias = self.r_emb[i], self.r_bias[i]

        mems_i = None if mems is None else mems[i]
        core_out = layer(core_out, r_emb, self.r_w_bias[i],
                         r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
        hids.append(core_out)
elif self.attn_type == 2: # absolute
    pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                           dtype=word_emb.dtype)
    if self.clamp_len > 0:
        pos_seq.clamp_(max=self.clamp_len)
    pos_emb = self.pos_emb(pos_seq)

    core_out = self.drop(word_emb + pos_emb[-qlen:])

    hids.append(core_out)
    for i, layer in enumerate(self.layers):
        mems_i = None if mems is None else mems[i]
        if mems_i is not None and i == 0:
            mems_i += pos_emb[:mlen]
        core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                         mems=mems_i)
        hids.append(core_out)
elif self.attn_type == 3:
    core_out = self.drop(word_emb)

    hids.append(core_out)
    for i, layer in enumerate(self.layers):
        mems_i = None if mems is None else mems[i]
        if mems_i is not None and mlen > 0:
            cur_emb = self.r_emb[i][:-qlen]
            cur_size = cur_emb.size(0)
            if cur_size < mlen:
                cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
                cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
            else:
                cur_emb = cur_emb[-mlen:]
            mems_i += cur_emb.view(mlen, 1, -1)
        core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

        core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                         mems=mems_i)
        hids.append(core_out)

core_out = self.drop(core_out)

new_mems = self._update_mems(hids, mems, mlen, qlen)

return core_out, new_mems

def forward(self, data, target, *mems):
    # nn.DataParallel does not allow size(0) tensors to be broadcasted.
    # So, have to initialize size(0) mems inside the model forward.
    # Moreover, have to return new_mems to allow nn.DataParallel to piece
    # them together.
    if not mems: mems = self.init_mems()

    tgt_len = target.size(0)
    hidden, new_mems = self._forward(data, mems=mems)

    pred_hid = hidden[-tgt_len:]
    if self.sample_softmax > 0 and self.training:
        assert self.tie_weight
        logit = sample_logits(self.word_emb,
                              self.out_layer.bias, target, pred_hid, self.sampler)
        loss = -F.log_softmax(logit, -1)[:, :, 0]
    else:
        loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
        loss = loss.view(tgt_len, -1)

    if new_mems is None:
        return [loss]
    else:
        return [loss] + new_mems