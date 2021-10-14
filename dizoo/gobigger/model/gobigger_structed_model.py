import torch
import torch.nn as nn
from ding.torch_utils import MLP, get_lstm, Transformer
from ding.model import DiscreteHead
from ding.utils import list_split


class GoBiggerStructedNetwork(nn.Module):

    def __init__(
            self,
            scalar_shape: int,
            per_unit_shape: int,
            action_type_shape: int,
            encoder_output_size_list=[128, 128],
            rnn: bool = False,
            activation=nn.ReLU(),
    ) -> None:
        super(GoBiggerStructedNetwork, self).__init__()
        self.activation = activation
        self.action_type_shape = action_type_shape

        scalar_output_size, unit_output_size = encoder_output_size_list
        assert unit_output_size == 128, "Not Implemented"
        # encoder
        self.scalar_encoder = MLP(
            scalar_shape, scalar_output_size, scalar_output_size, layer_num=2, activation=self.activation
        )
        self.unit_encoder = Transformer(
            per_unit_shape, head_dim=64, hidden_dim=128, output_dim=unit_output_size, activation=self.activation
        )
        # rnn
        if rnn:
            self.rnn = get_lstm('gru', input_size=sum(encoder_output_size_list), hidden_size=128)
        else:
            self.rnn = None
            self.main = MLP(sum(encoder_output_size_list), 128, 128, layer_num=2, activation=self.activation)
        # head
        self.action_type_head = DiscreteHead(128, action_type_shape, layer_num=2, activation=self.activation)

    def forward(self, obs: list) -> dict:
        """
            scalar_obs: torch.Tensor,
            unit_obs: torch.Tensor,
            prev_state: list = None,
        """
        B = obs[0]['scalar_obs'].shape[0]
        A = len(obs)
        # aggreate agent dim
        scalar_obs = torch.stack([o['scalar_obs'] for o in obs], dim=1)
        scalar_obs = scalar_obs.view(B * A, *scalar_obs.shape[2:])
        unit_obs_item = [o['unit_obs'] for o in obs]
        min_unit_num = min([t.shape[1] for t in unit_obs_item])
        unit_obs_item = [t[:, -min_unit_num:] for t in unit_obs_item]  # cutoff
        unit_obs = torch.stack(unit_obs_item, dim=1)
        unit_obs = unit_obs.view(B * A, *unit_obs.shape[2:])

        scalar_feature = self.scalar_encoder(scalar_obs)  # B, N1
        unit_feature = self.unit_encoder(unit_obs)  # B, M, N2

        fused_embedding_total = torch.cat([scalar_feature, unit_feature.mean(dim=1)], dim=1)
        if self.rnn:
            prev_state = []
            for b in range(B):
                for a in range(A):
                    prev_state.append(obs[a]['prev_state'][b])  # B, A, hidden_state_dim
            rnn_input = fused_embedding_total.unsqueeze(0)
            rnn_output, next_state = self.rnn(rnn_input, prev_state)
            next_state, _ = list_split(next_state, step=A)
            main_output = rnn_output.squeeze(0)  # B, N2
        else:
            main_output = self.main(fused_embedding_total)  # B, N2

        action_type_logit = self.action_type_head(main_output)['logit']  # B, M, action_type_size
        action_type_logit = action_type_logit.reshape(B, A, *action_type_logit.shape[1:])

        result = {
            'logit': action_type_logit,
        }
        if self.rnn:
            result['next_state'] = next_state
        return result
