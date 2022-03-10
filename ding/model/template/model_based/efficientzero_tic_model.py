import math
import torch

import numpy as np
import torch.nn as nn

from efficientzero_base_model import BaseNet, renormalize

## this is for tic-tac-toc model
## input size (3,3,3)  ->  reshape (27,)

def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=nn.Identity,
    activation=nn.ReLU,
    momentum=0.1,
    init_zero=False,
):
    """MLP layers
    Parameters
    ----------
    input_size: int
        dim of inputs
    layer_sizes: list
        dim of hidden layers
    output_size: int
        dim of outputs
    init_zero: bool
        zero initialization for the last layer (including w and b).
        This can provide stable zero outputs in the beginning.
    """
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       nn.BatchNorm1d(sizes[i + 1], momentum=momentum),
                       act()]
        else:
            act = output_activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       act()]

    if init_zero:
        layers[-2].weight.data.fill_(0)
        layers[-2].bias.data.fill_(0)

    return nn.Sequential(*layers)

# test mlp
# fc = mlp(27, [64,32,16,16], 9)
# print(fc)


# Predict next hidden states given current states and actions
class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        fc_reward_layers,
        full_support_size,
        block_output_size_reward,
        lstm_hidden_size=64,
        momentum=0.1,
        init_zero=False,
    ):
        """Dynamics network
        Parameters
        ----------
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        fc_reward_layers: list
            hidden layers of the reward prediction head (MLP head)
        full_support_size: int
            dim of reward output
        block_output_size_reward: int
            dim of flatten hidden states
        lstm_hidden_size: int
            dim of lstm hidden
        init_zero: bool
            True -> zero initialization for the last layer of reward mlp
        """
        super().__init__()

        self.lstm_hidden_size = lstm_hidden_size

        self.block_output_size_reward = block_output_size_reward
        self.lstm = nn.LSTM(input_size=self.block_output_size_reward, hidden_size=self.lstm_hidden_size)
        self.bn_value_prefix = nn.BatchNorm1d(self.lstm_hidden_size, momentum=momentum)
        self.fc = mlp(self.lstm_hidden_size, fc_reward_layers, full_support_size, init_zero=init_zero, momentum=momentum)

    def forward(self, x, reward_hidden):
        # state = x[:,:-1,:,:]  # todo: not know the atari state or tic-tac-toc state
        state = x

        x = x.view(-1, self.block_output_size_reward).unsqueeze(0)
        value_prefix, reward_hidden = self.lstm(x, reward_hidden)
        value_prefix = value_prefix.squeeze(0)
        value_prefix = self.bn_value_prefix(value_prefix)
        value_prefix = nn.functional.relu(value_prefix)
        value_prefix = self.fc(value_prefix)

        return state, reward_hidden, value_prefix

    def get_dynamic_mean(self):
        return None

    def get_reward_mean(self):
        for name, param in self.fc.named_parameters():
            temp_weights = param.detach().cpu().numpy().reshape(-1)
        reward_mean = np.abs(temp_weights).mean()
        return reward_mean


# predict the value and policy given hidden states
class PredictionNetwork(nn.Module):
    def __init__(
        self,
        action_space_size,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        input_size,
        momentum=0.1,
        init_zero=False,
    ):
        """Prediction network
        Parameters
        ----------
        action_space_size: int
            action space
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        reduced_channels_value: int
            channels of value head
        reduced_channels_policy: int
            channels of policy head
        fc_value_layers: list
            hidden layers of the value prediction head (MLP head)
        fc_policy_layers: list
            hidden layers of the policy prediction head (MLP head)
        full_support_size: int
            dim of value output
        block_output_size_value: int
            dim of flatten hidden states
        block_output_size_policy: int
            dim of flatten hidden states
        init_zero: bool
            True -> zero initialization for the last layer of value/policy mlp
        """
        super().__init__()

        self.fc_value = mlp(input_size, fc_value_layers, full_support_size, init_zero=init_zero, momentum=momentum)
        self.fc_policy = mlp(input_size, fc_policy_layers, action_space_size, init_zero=init_zero, momentum=momentum)

    def forward(self, x):
        # todo: x = x.reshape(:,input_size)

        value = self.fc_value(x)
        policy = self.fc_policy(x)
        return policy, value


class EfficientZeroNet(BaseNet):
    def __init__(
        self,
        observation_shape,
        action_space_size,

        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        reward_support_size,
        value_support_size,
        input_size,
        inverse_value_transform,
        inverse_reward_transform,
        lstm_hidden_size,
        bn_mt=0.1,
        proj_hid=256,
        proj_out=256,
        pred_hid=64,
        pred_out=256,
        init_zero=False,
        state_norm=False
    ):
        """EfficientZero network
        Parameters
        ----------
        observation_shape: tuple or list
            shape of observations: [C, W, H]
        action_space_size: int
            action space
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        reduced_channels_reward: int
            channels of reward head
        reduced_channels_value: int
            channels of value head
        reduced_channels_policy: int
            channels of policy head
        fc_reward_layers: list
            hidden layers of the reward prediction head (MLP head)
        fc_value_layers: list
            hidden layers of the value prediction head (MLP head)
        fc_policy_layers: list
            hidden layers of the policy prediction head (MLP head)
        reward_support_size: int
            dim of reward output
        value_support_size: int
            dim of value output
        downsample: bool
            True -> do downsampling for observations. (For board games, do not need)
        inverse_value_transform: Any
            A function that maps value supports into value scalars
        inverse_reward_transform: Any
            A function that maps reward supports into value scalars
        lstm_hidden_size: int
            dim of lstm hidden
        bn_mt: float
            Momentum of BN
        proj_hid: int
            dim of projection hidden layer
        proj_out: int
            dim of projection output layer
        pred_hid: int
            dim of projection head (prediction) hidden layer
        pred_out: int
            dim of projection head (prediction) output layer
        init_zero: bool
            True -> zero initialization for the last layer of value/policy mlp
        state_norm: bool
            True -> normalization for hidden states
        """
        super(EfficientZeroNet, self).__init__(inverse_value_transform, inverse_reward_transform, lstm_hidden_size)
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.init_zero = init_zero
        self.state_norm = state_norm

        self.action_space_size = action_space_size


        self.dynamics_network = DynamicsNetwork(
            fc_reward_layers,
            reward_support_size,
            lstm_hidden_size=lstm_hidden_size,
            momentum=bn_mt,
            init_zero=self.init_zero,
        )

        self.prediction_network = PredictionNetwork(
            action_space_size,
            fc_value_layers,
            fc_policy_layers,
            value_support_size,
            input_size,
            momentum=bn_mt,
            init_zero=self.init_zero,
        )

        # projection
        self.porjection_in_dim = observation_shape  # just for tic-tac-toe
        self.projection = nn.Sequential(
            nn.Linear(self.porjection_in_dim, self.proj_hid),
            nn.BatchNorm1d(self.proj_hid),
            nn.ReLU(),
            nn.Linear(self.proj_hid, self.proj_hid),
            nn.BatchNorm1d(self.proj_hid),
            nn.ReLU(),
            nn.Linear(self.proj_hid, self.proj_out),
            nn.BatchNorm1d(self.proj_out)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.proj_out, self.pred_hid),
            nn.BatchNorm1d(self.pred_hid),
            nn.ReLU(),
            nn.Linear(self.pred_hid, self.pred_out),
        )

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def dynamics(self, encoded_state, reward_hidden, action):
        # Stack encoded_state with a game specific one hot encoded action
        action_one_hot = (
            torch.ones(
                (
                    encoded_state.shape[0],
                    1,
                    encoded_state.shape[2],
                    encoded_state.shape[3],
                )
            )
            .to(action.device)
            .float()
        )
        action_one_hot = (
            action[:, :, None, None] * action_one_hot / self.action_space_size
        )
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, reward_hidden, value_prefix = self.dynamics_network(x, reward_hidden)

        if not self.state_norm:
            return next_encoded_state, reward_hidden, value_prefix
        else:
            next_encoded_state_normalized = renormalize(next_encoded_state)
            return next_encoded_state_normalized, reward_hidden, value_prefix

    def get_params_mean(self):
        dynamic_mean = self.dynamics_network.get_dynamic_mean()
        reward_mean = self.dynamics_network.get_reward_mean()

        return dynamic_mean, reward_mean

    def project(self, hidden_state, with_grad=True):
        # only the branch of proj + pred can share the gradients
        hidden_state = hidden_state.view(-1, self.porjection_in_dim)
        proj = self.projection(hidden_state)

        # with grad, use proj_head
        if with_grad:
            proj = self.projection_head(proj)
            return proj
        else:
            return proj.detach()

