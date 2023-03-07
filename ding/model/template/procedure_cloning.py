from typing import Optional, Tuple
import torch
import torch.nn as nn
from ding.utils import MODEL_REGISTRY, SequenceType
from ding.torch_utils.network.transformer import Attention
from ding.torch_utils.network.nn_module import fc_block, build_normalization
from ..common import FCEncoder, ConvEncoder


class Block(nn.Module):

    def __init__(
            self, cnn_hidden: int, att_hidden: int, att_heads: int, drop_p: float, max_T: int, n_att: int,
            feedforward_hidden: int, n_feedforward: int
    ) -> None:
        super().__init__()
        self.n_att = n_att
        self.n_feedforward = n_feedforward
        self.attention_layer = []

        self.norm_layer = [nn.LayerNorm(att_hidden)] * n_att
        self.attention_layer.append(Attention(cnn_hidden, att_hidden, att_hidden, att_heads, nn.Dropout(drop_p)))
        for i in range(n_att - 1):
            self.attention_layer.append(Attention(att_hidden, att_hidden, att_hidden, att_heads, nn.Dropout(drop_p)))

        self.att_drop = nn.Dropout(drop_p)

        self.fc_blocks = []
        self.fc_blocks.append(fc_block(att_hidden, feedforward_hidden, activation=nn.ReLU()))
        for i in range(n_feedforward - 1):
            self.fc_blocks.append(fc_block(feedforward_hidden, feedforward_hidden, activation=nn.ReLU()))
        self.norm_layer.extend([nn.LayerNorm(feedforward_hidden)] * n_feedforward)
        self.mask = torch.tril(torch.ones((max_T, max_T), dtype=torch.bool)).view(1, 1, max_T, max_T)

    def forward(self, x: torch.Tensor):
        for i in range(self.n_att):
            x = self.att_drop(self.attention_layer[i](x, self.mask))
            x = self.norm_layer[i](x)
        for i in range(self.n_feedforward):
            x = self.fc_blocks[i](x)
            x = self.norm_layer[i + self.n_att](x)
        return x


@MODEL_REGISTRY.register('pc')
class ProcedureCloning(nn.Module):

    def __init__(
            self,
            obs_shape: SequenceType,
            action_dim: int,
            cnn_hidden_list: SequenceType = [128, 128, 256, 256, 256],
            cnn_activation: Optional[nn.Module] = nn.ReLU(),
            cnn_kernel_size: SequenceType = [3, 3, 3, 3, 3],
            cnn_stride: SequenceType = [1, 1, 1, 1, 1],
            cnn_padding: Optional[SequenceType] = ['same', 'same', 'same', 'same', 'same'],
            mlp_hidden_list: SequenceType = [256, 256],
            mlp_activation: Optional[nn.Module] = nn.ReLU(),
            att_heads: int = 8,
            att_hidden: int = 128,
            n_att: int = 4,
            n_feedforward: int = 2,
            feedforward_hidden: int = 256,
            drop_p: float = 0.5,
            augment: bool = True,
            max_T: int = 17
    ) -> None:
        super().__init__()

        #Conv Encoder
        self.embed_state = ConvEncoder(
            obs_shape, cnn_hidden_list, cnn_activation, cnn_kernel_size, cnn_stride, cnn_padding
        )
        self.embed_action = FCEncoder(action_dim, mlp_hidden_list, activation=mlp_activation)

        self.cnn_hidden_list = cnn_hidden_list
        self.augment = augment

        assert cnn_hidden_list[-1] == mlp_hidden_list[-1]
        layers = []
        for i in range(n_att):
            if i == 0:
                layers.append(Attention(cnn_hidden_list[-1], att_hidden, att_hidden, att_heads, nn.Dropout(drop_p)))
            else:
                layers.append(Attention(att_hidden, att_hidden, att_hidden, att_heads, nn.Dropout(drop_p)))
            layers.append(build_normalization('LN')(att_hidden))
        for i in range(n_feedforward):
            if i == 0:
                layers.append(fc_block(att_hidden, feedforward_hidden, activation=nn.ReLU()))
            else:
                layers.append(fc_block(feedforward_hidden, feedforward_hidden, activation=nn.ReLU()))
                self.layernorm2 = build_normalization('LN')(feedforward_hidden)

        self.transformer = Block(
            cnn_hidden_list[-1], att_hidden, att_heads, drop_p, max_T, n_att, feedforward_hidden, n_feedforward
        )

        self.predict_goal = torch.nn.Linear(cnn_hidden_list[-1], cnn_hidden_list[-1])
        self.predict_action = torch.nn.Linear(cnn_hidden_list[-1], action_dim)

    def forward(self, states: torch.Tensor, goals: torch.Tensor,
                actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, T, _ = actions.shape

        # shape: (B, h_dim)
        state_embeddings = self.embed_state(states).reshape(B, 1, self.cnn_hidden_list[-1])
        goal_embeddings = self.embed_state(goals).reshape(B, 1, self.cnn_hidden_list[-1])
        # shape: (B, context_len, h_dim)
        actions_embeddings = self.embed_action(actions)

        h = torch.cat((state_embeddings, goal_embeddings, actions_embeddings), dim=1)
        h = self.transformer(h)
        h = h.reshape(B, T + 2, self.cnn_hidden_list[-1])

        goal_preds = self.predict_goal(h[:, 0, :])
        action_preds = self.predict_action(h[:, 1:, :])

        return goal_preds, action_preds
