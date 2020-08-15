import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import numpy as np
import math, random

from collections import namedtuple

from nervex.utils.log_helper import build_logger, build_logger_naive, pretty_print

from nervex.worker.actor.env_manager.base_env_manager import BaseEnvManager
from nervex.worker.actor.env_manager.vec_env_manager import SubprocessEnvManager

from nervex.data.structure.buffer import PrioritizedBuffer

from nervex.rl_utils.algorithms.dqnloss import DqnLoss

class DqnCNNnetwork(nn.Module):
    def __init__(self, h, w, outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.head(x.view(x.size(0), -1))


class DqnLoss(nn.Module):

    def __init__(self,
                discount_factor: Optional[float] = 0.99,
                q_function_criterion=nn.MSELoss()
        ):
        super().__init__()
        self._gamma = discount_factor
        self.q_function_criterion = q_function_criterion
        
  

    def forward(self, is_double, q_value, next_q_value ,target_q_value,  action,  reward, terminals, weights=None):
        rewards =  reward
        q_s_a = q_value.gather(1, action.unsqueeze(1).long()).squeeze(1)
        target_q_s_a = rewards + self._gamma * ( 1 - terminals) * target_q_value.gather(1, torch.max(next_q_value,1)[1].unsqueeze(1)).squeeze(1).to(device)
        if weights != None:
            q_function_loss = self.q_function_criterion(q_s_a, target_q_s_a.detach()) * weights
        else:
            q_function_loss = self.q_function_criterion(q_s_a, target_q_s_a.detach())
        return q_function_loss


            



class DqnRunner(nn.Module):
    r"""
    Overview:
        The Dqn Algorithnm runner
    Interface:
        __init__, train
    """

    def __init__(self, 
        q_network,
        total_frame_num: Optional[int] = 10000,
        is_dobule: Optional[bool] = False,
        prior_alpha: Optional[float] = 0.,
        buffer_len: Optional[int] = 1000,
        bandit: Optional[function] = None,

    ):
        r"""
        Arguments:
            - q_network,
            - isdouble,
            - usePri,
            - bandit (:obj:`function`)

        """
        self.q_function = q_network
        self.total_frame_num = total_frame_num
        self.is_dobule = is_dobule
        self.prior_alpha = prior_alpha
        
        self.buffer = PrioritizedBuffer(buffer_len, max_reuse=None, min_sample_ratio=1., alpha=prior_alpha, beta=0.)
        if bandit == None:
            self.bandit = lambda x: return 0.3
        else:
            self.bandit = bandit
        
        