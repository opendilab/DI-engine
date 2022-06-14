"""
The code is transplanted from https://github.com/nikhilbarhate99/min-decision-transformer
"""

from ding.utils import MODEL_REGISTRY
from typing import Tuple
from ding.torch_utils.network.transformer import Attention
import torch
import torch.nn as nn


class Block(nn.Module):

    def __init__(self, h_dim: int, max_T: int, n_heads: int, drop_p: float) -> None:
        super().__init__()
        self.attention = Attention(h_dim, h_dim, h_dim, n_heads, nn.Dropout(drop_p))
        self.att_drop = nn.Dropout(drop_p)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4 * h_dim),
            nn.GELU(),
            nn.Linear(4 * h_dim, h_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

        mask = torch.tril(torch.ones((max_T, max_T), dtype=torch.bool)).view(1, 1, max_T, max_T)
        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor):
        # Attention -> LayerNorm -> MLP -> LayerNorm

        x = x + self.att_drop(self.attention(x, self.mask))  # residual
        x = self.ln1(x)
        x = x + self.mlp(x)  # residual
        x = self.ln2(x)
        return x


@MODEL_REGISTRY.register('dt')
class DecisionTransformer(nn.Module):

    def __init__(
            self,
            state_dim: int,
            act_dim: int,
            n_blocks: int,
            h_dim: int,
            context_len: int,
            n_heads: int,
            drop_p: float,
            max_timestep: int = 4096,
            continuous: bool = True
    ) -> None:
        super().__init__()
        self.continuous = continuous
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        # transformer blocks
        # we will serially arrange `return`, `state` and `action`, so here the input_seq_len is 3 * context_len
        input_seq_len = 3 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        # projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        if self.continuous:
            action_tanh = True  # True for continuous actions
            self.embed_action = torch.nn.Linear(act_dim, h_dim)

        else:
            action_tanh = False  # False for discrete actions
            self.embed_action = torch.nn.Linear(act_dim, h_dim)

        # prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(*([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if action_tanh else [])))

    def forward(
            self, timesteps: torch.Tensor, states: torch.Tensor, actions: torch.Tensor, returns_to_go: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)  # shape: (B,context_len/T,h_dim)

        # time embeddings are treated similar to positional embeddings
        # shape: (B,context_len,h_dim)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings

        # stack rtg, states and actions and reshape sequence as
        # (r1, s1, a1, r2, s2, a2 ...)
        # after stack shape: (B, 3, context_len/T, h_dim)
        h = torch.stack((returns_embeddings, state_embeddings, action_embeddings),
                        dim=1).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t, a_t
        h = h.reshape(B, T, 3, self.h_dim)

        # get predictions
        return_preds = self.predict_rtg(h[..., 2, :])  # predict next rtg given r, s, a
        state_preds = self.predict_state(h[..., 2, :])  # predict next state given r, s, a
        action_preds = self.predict_action(h[..., 1, :])  # predict action given r, s

        return state_preds, action_preds, return_preds
