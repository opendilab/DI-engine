"""
this extremely minimal Decision Transformer model is based on
the following causal transformer (GPT) implementation:

Misha Laskin's tweet:
https://twitter.com/MishaLaskin/status/1481767788775628801?cxt=HHwWgoCzmYD9pZApAAAA

and its corresponding notebook:
https://colab.research.google.com/drive/1NUBqyboDcGte5qAJKOl8gaJC28V_73Iv?usp=sharing

** the above colab notebook has a bug while applying masked_fill
which is fixed in the following code
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCausalAttention(nn.Module):

    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape  # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads  # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[..., :T, :T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):

    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4 * h_dim),
            nn.GELU(),
            nn.Linear(4 * h_dim, h_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x)  # residual
        x = self.ln1(x)
        x = x + self.mlp(x)  # residual
        x = self.ln2(x)
        # x = x + self.attention(self.ln1(x))
        # x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):

    def __init__(
        self,
        state_dim,
        act_dim,
        n_blocks,
        h_dim,
        context_len,
        n_heads,
        drop_p,
        max_timestep=4096,
        state_encoder=None,
        continuous=False
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        # transformer blocks
        input_seq_len = 3 * context_len

        # projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.drop = nn.Dropout(drop_p)

        self.pos_emb = nn.Parameter(torch.zeros(1, input_seq_len + 1, self.h_dim))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, max_timestep + 1, self.h_dim))

        if state_encoder is None:
            self.state_encoder = None
            blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
            self.embed_rtg = torch.nn.Linear(1, h_dim)
            self.embed_state = torch.nn.Linear(state_dim, h_dim)
            self.predict_rtg = torch.nn.Linear(h_dim, 1)
            self.predict_state = torch.nn.Linear(h_dim, state_dim)
            if continuous:
                # continuous actions
                self.embed_action = torch.nn.Linear(act_dim, h_dim)
                use_action_tanh = True  # True for continuous actions
            else:
                # discrete actions
                self.embed_action = torch.nn.Embedding(act_dim, h_dim)
                use_action_tanh = False  # False for discrete actions
            self.predict_action = nn.Sequential(
                *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
            )
        else:
            blocks = [Block(h_dim, input_seq_len + 1, n_heads, drop_p) for _ in range(n_blocks)]
            self.state_encoder = state_encoder
            self.embed_rtg = nn.Sequential(nn.Linear(1, h_dim), nn.Tanh())
            self.head = nn.Linear(h_dim, act_dim, bias=False)
            self.embed_action = nn.Sequential(nn.Embedding(act_dim, h_dim), nn.Tanh())
        self.transformer = nn.Sequential(*blocks)

    def forward(self, timesteps, states, actions, returns_to_go, tar=None):
        B, T = states.shape[0], states.shape[1]
        if self.state_encoder is None:
            time_embeddings = self.embed_timestep(timesteps)

            # time embeddings are treated similar to positional embeddings
            state_embeddings = self.embed_state(states) + time_embeddings
            action_embeddings = self.embed_action(actions) + time_embeddings
            returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings

            # stack rtg, states and actions and reshape sequence as
            # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
            t_p = torch.stack((returns_embeddings, state_embeddings, action_embeddings),
                              dim=1).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)
            h = self.embed_ln(t_p)
            # transformer and prediction
            h = self.transformer(h)
            # get h reshaped such that its size = (B x 3 x T x h_dim) and
            # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
            # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
            # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
            # that is, for each timestep (t) we have 3 output embeddings from the transformer,
            # each conditioned on all previous timesteps plus
            # the 3 input variables at that timestep (r_t, s_t, a_t) in sequence.
            h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)

            return_preds = self.predict_rtg(h[:, 2])  # predict next rtg given r, s, a
            state_preds = self.predict_state(h[:, 2])  # predict next state given r, s, a
            action_preds = self.predict_action(h[:, 1])  # predict action given r, s
        else:
            state_embeddings = self.state_encoder(
                states.reshape(-1, *self.state_dim).type(torch.float32).contiguous()
            )  # (batch * block_size, h_dim)
            state_embeddings = state_embeddings.reshape(B, T, self.h_dim)  # (batch, block_size, h_dim)
            returns_embeddings = self.embed_rtg(returns_to_go.type(torch.float32))
            action_embeddings = self.embed_action(actions.type(torch.long).squeeze(-1))  # (batch, block_size, h_dim)

            token_embeddings = torch.zeros(
                (B, T * 3 - int(tar is None), self.h_dim), dtype=torch.float32, device=state_embeddings.device
            )
            token_embeddings[:, ::3, :] = returns_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings[:, -T + int(tar is None):, :]

            all_global_pos_emb = torch.repeat_interleave(
                self.global_pos_emb, B, dim=0
            )  # batch_size, traj_length, h_dim

            position_embeddings = torch.gather(
                all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.h_dim, dim=-1)
            ) + self.pos_emb[:, :token_embeddings.shape[1], :]

            t_p = token_embeddings + position_embeddings

            h = self.drop(t_p)
            h = self.transformer(h)
            h = self.embed_ln(h)
            logits = self.head(h)

            return_preds = None
            state_preds = None
            action_preds = logits[:, 1::3, :]  # only keep predictions from state_embeddings

        return state_preds, action_preds, return_preds

    def configure_optimizers(self, weight_decay, learning_rate, betas=(0.9, 0.95)):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0,\
            "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0
            },
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
