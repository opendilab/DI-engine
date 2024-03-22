from random import random
from functools import partial, cache

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList

from beartype import beartype
from beartype.typing import Union, List, Optional, Callable, Tuple, Dict, Any

from einops import pack, unpack, repeat, reduce, rearrange
from einops.layers.torch import Rearrange, Reduce
from functools import wraps
from packaging import version

from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
# from q_transformer.attend import Attend


#myself code of xue 
class state_encode(nn.Module):
    def __init__(self, input_dim):
        super(state_encode, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.unsqueeze(1)
        return x

def exists(val):
    return val is not None

def xnor(x, y):
    """ (True, True) or (False, False) -> True """
    return not (x ^ y)

def divisible_by(num, den):
    return (num % den) == 0

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)


def l2norm(t, dim = -1):
    return F.normalize(t, dim = dim)

def pack_one(x, pattern):
    return pack([x], pattern)

def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]


class RMSNorm(Module):
    def __init__(self, dim, affine = True):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if affine else 1.

    def forward(self, x):
        return l2norm(x) * self.gamma * self.scale

class ChanRMSNorm(Module):
    def __init__(self, dim, affine = True):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1)) if affine else 1.

    def forward(self, x):
        return l2norm(x, dim = 1) * self.gamma * self.scale



class FeedForward(Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.,
        adaptive_ln = False
    ):
        super().__init__()
        self.adaptive_ln = adaptive_ln

        inner_dim = int(dim * mult)
        self.norm = RMSNorm(dim, affine = not adaptive_ln)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        cond_fn: Optional[Callable] = None
    ):
        x = self.norm(x)

        assert xnor(self.adaptive_ln, exists(cond_fn))

        if exists(cond_fn):
            # adaptive layernorm
            x = cond_fn(x)

        return self.net(x)


class TransformerAttention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        dim_context = None,
        heads = 8,
        num_mem_kv = 4,
        norm_context = False,
        adaptive_ln = False,
        dropout = 0.1,
        flash = True,
        causal = False
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.adaptive_ln = adaptive_ln
        self.norm = RMSNorm(dim, affine = not adaptive_ln)

        self.context_norm = RMSNorm(dim_context) if norm_context else None

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias = False)

        self.num_mem_kv = num_mem_kv
        self.mem_kv = None
        if num_mem_kv > 0:
            self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))

        self.attend = Attend(
            dropout = dropout,
            flash = flash,
            causal = causal
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_mask = None,
        cond_fn: Optional[Callable] = None,
        cache: Optional[Tensor] = None,
        return_cache = False
    ):
        b = x.shape[0]

        assert xnor(exists(context), exists(self.context_norm))

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        assert xnor(exists(cond_fn), self.adaptive_ln)

        if exists(cond_fn):
            x = cond_fn(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        if exists(cache):
            ck, cv = cache
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        new_kv_cache = torch.stack((k, v))

        if exists(self.mem_kv):
            mk, mv = map(lambda t: repeat(t, '... -> b ...', b = b), self.mem_kv)

            k = torch.cat((mk, k), dim = -2)
            v = torch.cat((mv, v), dim = -2)

            if exists(mask):
                mask = F.pad(mask, (self.num_mem_kv, 0), value = True)

            if exists(attn_mask):
                attn_mask = F.pad(attn_mask, (self.num_mem_kv, 0), value = True)

        out = self.attend(q, k, v, mask = mask, attn_mask = attn_mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if not return_cache:
            return out

        return out, new_kv_cache

class Transformer(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        depth = 6,
        attn_dropout = 0.,
        ff_dropout = 0.,
        adaptive_ln = False,
        flash_attn = True,
        cross_attend = False,
        causal = False,
        final_norm = True
    ):
        super().__init__()
        self.layers = ModuleList([])

        attn_kwargs = dict(
            dim = dim,
            heads = heads,
            dim_head = dim_head,
            dropout = attn_dropout,
            flash = flash_attn
        )

        for _ in range(depth):
            self.layers.append(ModuleList([
                TransformerAttention(**attn_kwargs, causal = causal, adaptive_ln = adaptive_ln, norm_context = False),
                TransformerAttention(**attn_kwargs, norm_context = True) if cross_attend else None,
                FeedForward(dim = dim, dropout = ff_dropout, adaptive_ln = adaptive_ln)
            ]))

        self.norm = RMSNorm(dim) if final_norm else nn.Identity()

    @beartype
    def forward(
        self,
        x,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        attn_mask = None,
        context: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
        return_cache = False
    ):
        has_cache = exists(cache)

        if has_cache:
            x_prev, x = x[..., :-1, :], x[..., -1:, :]

        cond_fns = iter(default(cond_fns, []))
        cache = iter(default(cache, []))

        new_caches = []

        for attn, maybe_cross_attn, ff in self.layers:
            attn_out, new_cache = attn(
                x,
                attn_mask = attn_mask,
                cond_fn = next(cond_fns, None),
                return_cache = True,
                cache = next(cache, None)
            )

            new_caches.append(new_cache)

            x = x + attn_out

            if exists(maybe_cross_attn):
                assert exists(context)
                x = maybe_cross_attn(x, context = context) + x

            x = ff(x, cond_fn = next(cond_fns, None)) + x

        new_caches = torch.stack(new_caches)

        if has_cache:
            x = torch.cat((x_prev, x), dim = -2)

        out = self.norm(x)

        if not return_cache:
            return out

        return out, new_caches
    


class DuelingHead(Module):
    def __init__(
        self,
        dim,
        expansion_factor = 2,
        action_bins = 256
    ):
        super().__init__()
        dim_hidden = dim * expansion_factor

        self.stem = nn.Sequential(
            nn.Linear(dim, dim_hidden),
            nn.SiLU()
        )

        self.to_values = nn.Sequential(
            nn.Linear(dim_hidden, 1)
        )

        self.to_advantages = nn.Sequential(
            nn.Linear(dim_hidden, action_bins)
        )

    def forward(self, x):
        x = self.stem(x)

        advantages = self.to_advantages(x)
        advantages = advantages - reduce(advantages, '... a -> ... 1', 'mean')

        values = self.to_values(x)

        q_values = values + advantages
        return q_values.sigmoid()


class QHeadSingleAction(Module):
    def __init__(
        self,
        dim,
        *,
        num_learned_tokens = 8,
        action_bins = 256,
        dueling = False
    ):
        super().__init__()
        self.action_bins = action_bins

        if dueling:
            self.to_q_values = nn.Sequential(
                Reduce('b (f n) d -> b d', 'mean', n = num_learned_tokens),
                DuelingHead(
                    dim,
                    action_bins = action_bins
                )
            )
        else:
            self.to_q_values = nn.Sequential(
                Reduce('b (f n) d -> b d', 'mean', n = num_learned_tokens),
                RMSNorm(dim),
                nn.Linear(dim, action_bins),
                nn.Sigmoid()
            )

    def get_random_actions(self, batch_size):
        return torch.randint(0, self.action_bins, (batch_size,), device = self.device)

    def get_optimal_actions(
        self,
        encoded_state,
        return_q_values = False,
        actions = None,
        **kwargs
    ):
        assert not exists(actions), 'single actions will never receive previous actions'

        q_values = self.forward(encoded_state)

        max_q, action_indices = q_values.max(dim = -1)

        if not return_q_values:
            return action_indices

        return action_indices, max_q

    def forward(self, encoded_state):
        return self.to_q_values(encoded_state)

class QHeadMultipleActions(Module):
    def __init__(
        self,
        dim,
        *,
        num_actions = 3,
        action_bins = 256,
        attn_depth = 2,
        attn_dim_head = 32,
        attn_heads = 8,
        dueling = False,
        weight_tie_action_bin_embed = False,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.action_bins = action_bins

        self.action_bin_embeddings = nn.Parameter(torch.zeros(num_actions, action_bins, dim))
        nn.init.normal_(self.action_bin_embeddings, std = 0.02)

        self.to_q_values = None
        if not weight_tie_action_bin_embed:
            self.to_q_values = nn.Linear(dim, action_bins)

        self.transformer = Transformer(
            dim = dim,
            depth = attn_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            cross_attend = True,
            adaptive_ln = False,
            causal = True,
            final_norm = True
        )

        self.final_norm = RMSNorm(dim)

        self.dueling = dueling
        if dueling:
            self.to_values = nn.Parameter(torch.zeros(num_actions, dim))

    @property
    def device(self):
        return self.action_bin_embeddings.device

    def maybe_append_actions(self, sos_tokens, actions: Optional[Tensor] = None):
        if not exists(actions):
            return sos_tokens

        batch, num_actions = actions.shape
        action_embeddings = self.action_bin_embeddings[:num_actions]

        action_embeddings = repeat(action_embeddings, 'n a d -> b n a d', b = batch)
        past_action_bins = repeat(actions, 'b n -> b n 1 d', d = action_embeddings.shape[-1])

        bin_embeddings = action_embeddings.gather(-2, past_action_bins)
        bin_embeddings = rearrange(bin_embeddings, 'b n 1 d -> b n d')

        tokens, _ = pack((sos_tokens, bin_embeddings), 'b * d')
        tokens = tokens[:, :self.num_actions] # last action bin not needed for the proposed q-learning
        return tokens

    def get_q_values(self, embed):
        num_actions = embed.shape[-2]

        if exists(self.to_q_values):
            logits = self.to_q_values(embed)
        else:
            # each token predicts next action bin
            action_bin_embeddings = self.action_bin_embeddings[:num_actions]
            action_bin_embeddings = torch.roll(action_bin_embeddings, shifts = -1, dims = 1)
            logits = einsum('b n d, n a d -> b n a', embed, action_bin_embeddings)

        if self.dueling:
            advantages = logits
            values = einsum('b n d, n d -> b n', embed, self.to_values[:num_actions])
            values = rearrange(values, 'b n -> b n 1')

            q_values = values + (advantages - reduce(advantages, '... a -> ... 1', 'mean'))
        else:
            q_values = logits

        return q_values.sigmoid()

    def get_random_actions(self, batch_size, num_actions = None):
        num_actions = default(num_actions, self.num_actions)
        return torch.randint(0, self.action_bins, (batch_size, num_actions), device = self.device)

    @torch.no_grad()
    def get_optimal_actions(
        self,
        encoded_state,
        return_q_values = False,
        actions: Optional[Tensor] = None,
        prob_random_action: float = 0.5,
        **kwargs
    ):
        assert 0. <= prob_random_action <= 1.
        batch = encoded_state.shape[0]

        if prob_random_action == 1:
            return self.get_random_actions(batch)

        sos_token = encoded_state
        tokens = self.maybe_append_actions(sos_token, actions = actions)

        action_bins = []
        cache = None

        for action_idx in range(self.num_actions):

            embed, cache = self.transformer(
                tokens,
                context = encoded_state,
                cache = cache,
                return_cache = True
            )

            last_embed = embed[:, action_idx]
            bin_embeddings = self.action_bin_embeddings[action_idx]

            q_values = einsum('b d, a d -> b a', last_embed, bin_embeddings)

            selected_action_bins = q_values.argmax(dim = -1)

            if prob_random_action > 0.:
                random_mask = torch.zeros_like(selected_action_bins).float().uniform_(0., 1.) < prob_random_action
                random_actions = self.get_random_actions(batch, 1)
                random_actions = rearrange(random_actions, '... 1 -> ...')

                selected_action_bins = torch.where(
                    random_mask,
                    random_actions,
                    selected_action_bins
                )

            next_action_embed = bin_embeddings[selected_action_bins]

            tokens, _ = pack((tokens, next_action_embed), 'b * d')

            action_bins.append(selected_action_bins)

        action_bins = torch.stack(action_bins, dim = -1)

        if not return_q_values:
            return action_bins

        all_q_values = self.get_q_values(embed)
        return action_bins, all_q_values

    def forward(
        self,
        encoded_state: Tensor,
        actions: Optional[Tensor] = None
    ):
        """
        einops
        b - batch
        n - number of actions
        a - action bins
        d - dimension
        """

        # this is the scheme many hierarchical transformer papers do
        tokens = encoded_state
        sos_token = encoded_state
        tokens = self.maybe_append_actions(sos_token, actions = actions)
        embed = self.transformer(tokens, context = encoded_state)
        return self.get_q_values(embed)

# Robotic Transformer
class QTransformer(Module):
    @beartype
    def __init__(
        self,
        num_actions = 3,
        action_bins = 256,
        depth = 6,
        heads = 8,
        dim_head = 64,
        obs_dim = 11,
        token_learner_ff_mult = 2,
        token_learner_num_layers = 2,
        token_learner_num_output_tokens = 8,
        cond_drop_prob = 0.2,
        use_attn_conditioner = False,
        conditioner_kwargs: dict = dict(),
        dueling = False,                       
        flash_attn = True,
        condition_on_text = True,
        q_head_attn_kwargs: dict = dict(
            attn_heads = 8,
            attn_dim_head = 64,
            attn_depth = 2
        ),
        weight_tie_action_bin_embed = True     
    ):
        super().__init__()
        attend_dim = 512
        # q-transformer related action embeddings
        assert num_actions >= 1
        self.num_actions = num_actions
        self.is_single_action = num_actions == 1
        self.action_bins = action_bins
        self.obs_dim = obs_dim

        #encode state
        self.state_encode =state_encode(self.obs_dim)

        # Q head
        if self.is_single_action:
            self.q_head = QHeadSingleAction(
                attend_dim,
                num_learned_tokens = self.num_learned_tokens,
                action_bins = action_bins,
                dueling = dueling
            )
        else:
            self.q_head = QHeadMultipleActions(
                attend_dim,
                action_bins = action_bins,
                dueling = dueling,
                weight_tie_action_bin_embed = weight_tie_action_bin_embed,
                **q_head_attn_kwargs
            )
    @property
    def device(self):
        return next(self.parameters()).device

    def get_random_actions(self, batch_size = 1):
        return self.q_head.get_random_actions(batch_size)

    @beartype
    def embed_texts(self, texts: List[str]):
        return self.conditioner.embed_texts(texts)

    @torch.no_grad()
    def get_optimal_actions(
        self,
        state,
        return_q_values = False,
        actions: Optional[Tensor] = None,
        **kwargs
    ):
        encoded_state = self.state_encode(state)
        return self.q_head.get_optimal_actions(encoded_state, return_q_values = return_q_values, actions = actions)

    def get_actions(
        self,
        state,
        prob_random_action = 0.,  # otherwise known as epsilon in RL
        **kwargs,
    ):
        batch_size = state.shape[0]
        assert 0. <= prob_random_action <= 1.

        if random() < prob_random_action:
            return self.get_random_actions(batch_size = batch_size)
        return self.get_optimal_actions(state, **kwargs)
    
    def forward(
            self,
            state: Tensor,
            actions: Optional[Tensor] = None,
            cond_drop_prob = 0.,
    ):
        state=state.to(self.device)
        if exists(actions):
            actions = actions.to(self.device)
        encoded_state = self.state_encode(state)
        if self.is_single_action:
            assert not exists(actions), 'actions should not be passed in for single action robotic transformer'
            q_values = self.q_head(encoded_state)
        else:
            q_values = self.q_head(encoded_state, actions = actions)
        return q_values





def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def maybe_reduce_mask_and(*maybe_masks):
    maybe_masks = [*filter(exists, maybe_masks)]

    if len(maybe_masks) == 0:
        return None

    mask, *rest_masks = maybe_masks

    for rest_mask in rest_masks:
        mask = mask & rest_mask

    return mask



# main class

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        causal = False,
        flash_config: dict = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        if flash:
            print_once('using memory efficient attention')

        self.flash_config = flash_config

    def flash_attn(self, q, k, v, mask = None, attn_mask = None):
        _, heads, q_len, dim_head, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if exists(mask):
            mask = mask.expand(-1, heads, q_len, -1)

        mask = maybe_reduce_mask_and(mask, attn_mask)

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale

        with torch.backends.cuda.sdp_kernel(**self.flash_config):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                is_causal = self.causal,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v, mask = None, attn_mask = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if exists(mask) and mask.ndim != 4:
            mask = rearrange(mask, 'b j -> b 1 1 j')

        if self.flash:
            return self.flash_attn(q, k, v, mask = mask, attn_mask = attn_mask)

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # causal mask

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = sim.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # key padding mask

        if exists(mask):
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # attention mask

        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
    
    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model with argmax strategy.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ReturnsKeys
            - necessary: ``action``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    
    