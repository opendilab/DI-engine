from torch import nn

from ding.torch_utils.network.nn_module import conv2d_block
from ding.torch_utils.network.res_block import ResBlock
from ding.utils.registry_factory import MODEL_REGISTRY


@MODEL_REGISTRY.register('tictactoe_model')
class TicTacToeModel(nn.Module):
    """policy-value network module"""

    def __init__(self, model_cfg={}):
        super(TicTacToeModel, self).__init__()
        self.cfg = model_cfg
        self.input_channels = 3
        self.board_width = 3
        self.board_height = 3

        # encoder part
        self.encoder = nn.Sequential(
            conv2d_block(in_channels=3,out_channels=16,kernel_size=1,stride=1,padding=0,activation=nn.ReLU(),norm_type=None),
            ResBlock(in_channels=16,activation=nn.ReLU(),norm_type=None)
        )

        # action policy head
        self.policy_head = nn.Sequential(
            conv2d_block(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()),
            nn.Flatten(),
            nn.Linear(4 * self.board_width * self.board_height, self.board_width * self.board_height)
        )

        # state value layers
        self.value_head = nn.Sequential(
            conv2d_block(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()),
            nn.Flatten(),
            nn.Linear(2 * self.board_width * self.board_height, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, state_input):
        # common layers
        encoded_embedding = self.encoder(state_input)
        # action policy layers
        logit = self.policy_head(encoded_embedding)
        # state value layers
        value = self.value_head(encoded_embedding)
        return logit, value


if __name__ == '__main__':
    import torch
    from easydict import EasyDict

    board_width = 3
    board_height = 3
    input_channels = 3
    batch_size = 3


    inputs = torch.randn(batch_size, input_channels, board_width, board_height)
    model = TicTacToeModel({})
    print(model(inputs))
