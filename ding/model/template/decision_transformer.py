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
from typing import Union, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.utils import SequenceType


class MaskedCausalAttention(nn.Module):
    """
    Overview:
        The implementation of masked causal attention in decision transformer. The input of this module is a sequence \
        of several tokens. For the calculated hidden embedding for the i-th token, it is only related the 0 to i-1 \
        input tokens by applying a mask to the attention map. Thus, this module is called masked-causal attention.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, h_dim: int, max_T: int, n_heads: int, drop_p: float) -> None:
        """
        Overview:
            Initialize the MaskedCausalAttention Model according to input arguments.
        Arguments:
            - h_dim (:obj:`int`): The dimension of the hidden layers, such as 128.
            - max_T (:obj:`int`): The max context length of the attention, such as 6.
            - n_heads (:obj:`int`): The number of heads in calculating attention, such as 8.
            - drop_p (:obj:`float`): The drop rate of the drop-out layer, such as 0.1.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            MaskedCausalAttention forward computation graph, input a sequence tensor \
            and return a tensor with the same shape.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - out (:obj:`torch.Tensor`): Output tensor, the shape is the same as the input.
        Examples:
            >>> inputs = torch.randn(2, 4, 64)
            >>> model = MaskedCausalAttention(64, 5, 4, 0.1)
            >>> outputs = model(inputs)
            >>> assert outputs.shape == torch.Size([2, 4, 64])
        """
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
    """
    Overview:
        The implementation of a transformer block in decision transformer.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, h_dim: int, max_T: int, n_heads: int, drop_p: float) -> None:
        """
        Overview:
            Initialize the Block Model according to input arguments.
        Arguments:
            - h_dim (:obj:`int`): The dimension of the hidden layers, such as 128.
            - max_T (:obj:`int`): The max context length of the attention, such as 6.
            - n_heads (:obj:`int`): The number of heads in calculating attention, such as 8.
            - drop_p (:obj:`float`): The drop rate of the drop-out layer, such as 0.1.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Forward computation graph of the decision transformer block, input a sequence tensor \
            and return a tensor with the same shape.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - output (:obj:`torch.Tensor`): Output tensor, the shape is the same as the input.
        Examples:
            >>> inputs = torch.randn(2, 4, 64)
            >>> model = Block(64, 5, 4, 0.1)
            >>> outputs = model(inputs)
            >>> outputs.shape == torch.Size([2, 4, 64])
        """
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x)  # residual
        x = self.ln1(x)
        x = x + self.mlp(x)  # residual
        x = self.ln2(x)
        # x = x + self.attention(self.ln1(x))
        # x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):
    """
    Overview:
        The implementation of decision transformer.
    Interfaces:
        ``__init__``, ``forward``, ``configure_optimizers``
    """

    def __init__(
        self,
        state_dim: Union[int, SequenceType],
        act_dim: int,
        n_blocks: int,
        h_dim: int,
        context_len: int,
        n_heads: int,
        drop_p: float,
        max_timestep: int = 4096,
        state_encoder: Optional[nn.Module] = None,
        continuous: bool = False
    ):
        """
        Overview:
            Initialize the DecisionTransformer Model according to input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Dimension of state, such as 128 or (4, 84, 84).
            - act_dim (:obj:`int`): The dimension of actions, such as 6.
            - n_blocks (:obj:`int`): The number of transformer blocks in the decision transformer, such as 3.
            - h_dim (:obj:`int`): The dimension of the hidden layers, such as 128.
            - context_len (:obj:`int`): The max context length of the attention, such as 6.
            - n_heads (:obj:`int`): The number of heads in calculating attention, such as 8.
            - drop_p (:obj:`float`): The drop rate of the drop-out layer, such as 0.1.
            - max_timestep (:obj:`int`): The max length of the total sequence, defaults to be 4096.
            - state_encoder (:obj:`Optional[nn.Module]`): The encoder to pre-process the given input. If it is set to \
                None, the raw state will be pushed into the transformer.
            - continuous (:obj:`bool`): Whether the action space is continuous, defaults to be ``False``.
        """
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

    def forward(
            self,
            timesteps: torch.Tensor,
            states: torch.Tensor,
            actions: torch.Tensor,
            returns_to_go: torch.Tensor,
            tar: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Overview:
            Forward computation graph of the decision transformer, input a sequence tensor \
            and return a tensor with the same shape.
        Arguments:
            - timesteps (:obj:`torch.Tensor`): The timestep for input sequence.
            - states (:obj:`torch.Tensor`): The sequence of states.
            - actions (:obj:`torch.Tensor`): The sequence of actions.
            - returns_to_go (:obj:`torch.Tensor`): The sequence of return-to-go.
            - tar (:obj:`Optional[int]`): Whether to predict action, regardless of index.
        Returns:
            - output (:obj:`Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`): Output contains three tensors, \
            they are correspondingly the predicted states, predicted actions and predicted return-to-go.
        Examples:
            >>> B, T = 4, 6
            >>> state_dim = 3
            >>> act_dim = 2
            >>> DT_model = DecisionTransformer(\
                state_dim=state_dim,\
                act_dim=act_dim,\
                n_blocks=3,\
                h_dim=8,\
                context_len=T,\
                n_heads=2,\
                drop_p=0.1,\
            )
            >>> timesteps = torch.randint(0, 100, [B, 3 * T - 1, 1], dtype=torch.long)  # B x T
            >>> states = torch.randn([B, T, state_dim])  # B x T x state_dim
            >>> actions = torch.randint(0, act_dim, [B, T, 1])
            >>> action_target = torch.randint(0, act_dim, [B, T, 1])
            >>> returns_to_go_sample = torch.tensor([1, 0.8, 0.6, 0.4, 0.2, 0.]).repeat([B, 1]).unsqueeze(-1).float()
            >>> traj_mask = torch.ones([B, T], dtype=torch.long)  # B x T
            >>> actions = actions.squeeze(-1)
            >>> state_preds, action_preds, return_preds = DT_model.forward(\
                timesteps=timesteps, states=states, actions=actions, returns_to_go=returns_to_go\
            )
            >>> assert state_preds.shape == torch.Size([B, T, state_dim])
            >>> assert return_preds.shape == torch.Size([B, T, 1])
            >>> assert action_preds.shape == torch.Size([B, T, act_dim])
        """
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

    def configure_optimizers(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float] = (0.9, 0.95)
    ) -> torch.optim.Optimizer:
        """
        Overview:
            This function returns an optimizer given the input arguments. \
            We are separating out all parameters of the model into two buckets: those that will experience \
            weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        Arguments:
            - weight_decay (:obj:`float`): The weigh decay of the optimizer.
            - learning_rate (:obj:`float`): The learning rate of the optimizer.
            - betas (:obj:`Tuple[float, float]`): The betas for Adam optimizer.
        Outputs:
            - optimizer (:obj:`torch.optim.Optimizer`): The desired optimizer.
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
