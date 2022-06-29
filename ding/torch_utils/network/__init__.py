from .activation import build_activation, Swish
from .res_block import ResBlock, ResFCBlock
from .nn_module import fc_block, conv2d_block, one_hot, deconv2d_block, BilinearUpsample, NearestUpsample, \
    binary_encode, NoiseLinearLayer, noise_block, MLP, Flatten, normed_linear, normed_conv2d
from .normalization import build_normalization
from .rnn import get_lstm, sequence_mask
from .soft_argmax import SoftArgmax
from .transformer import Transformer
from .scatter_connection import ScatterConnection
from .resnet import resnet18, ResNet
from .gumbel_softmax import GumbelSoftmax
from .gtrxl import GTrXL, GRUGatingUnit
