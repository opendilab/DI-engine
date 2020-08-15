import torch
import numpy as np
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from typing import Dict, Union, Optional
from nervex.data.structure.buffer import PrioritizedBuffer
import math
import random
from itertools import count
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DqnLoss(nn.Module):

    def __init__(self,
                discount_factor: Optional[float] = 0.99,
                q_function_criterion=nn.MSELoss()
        ):
        super().__init__()
        self._gamma = discount_factor
        # self.q_function = q_function
        # self.target_q_fuction = deepcopy(q_function)
        self.q_function_criterion = nn.MSELoss(reduce=True, size_average=True)
        
  

    def forward(self, q_value, next_q_value ,target_q_value,  action,  reward, terminals, weights=None):
        
        # reward     = Variable(torch.FloatTensor(reward))
        # weights    = Variable(torch.FloatTensor(weights))

        # rewards = Variable(reward)
        # terminals = Variable(terminals)
        rewards =  reward
        # print("target_qvalue = ", target_q_value)
        # print("terminals = ", terminals)
        # print("action indexes = ",action.unsqueeze(1).long())
        q_s_a = q_value.gather(1, action.unsqueeze(1).long()).squeeze(1)
        
        # print("q_value = ", q_value)
        # print("q_s_a = ", q_s_a)
        #target_q_s_a = rewards + self._gamma * ( 1 - terminals) * target_q_value.max(1, keepdim=False)[0].to(device)

        # print("indexs = ", torch.max(next_q_value,1)[1].unsqueeze(1))

        # target_q_value = 

        target_q_s_a = rewards + self._gamma * ( 1 - terminals) * target_q_value.gather(1, torch.max(next_q_value,1)[1].unsqueeze(1)).squeeze(1).to(device)
        # target_q_s_a = target_q_value.gather(1, torch.max(next_q_value,1)[1].unsqueeze(1)).to(device)
        # print(target_q_s_a)
        # print("rewards = ", rewards)
        # print('1- terminals =', 1- terminals)
        # print("* = ", ( 1 - terminals)*target_q_s_a.squeeze())
        # target_q_s_a = rewards + self._gamma * ( 1 - terminals) * target_q_value.max(1, keepdim=True)[0].to('cuda')
        # print("rewards = ", reward)
        # print("qvalue = ", q_value)
        # print("target_q_s_a = ", target_q_s_a)
        # if not terminals:
        #     target_q_s_a = rewards + self._gamma * ( 1 - terminals) * target_q_value.max(1, keepdim=True)[0]
        # else:
        #     target_q_s_a = rewards
        # print(q_value.shape)
        # print(target_q_s_a.shape)
        # target_q_s_a = target_q_s_a.unsqueeze(1)
        # target_q_s_a.expand(len(target_q_s_a), q_value.size()[1])
        # target_q_s_a.expand_as(q_s_a)
        # print("q_s_a = ", q_s_a)
        # print("target_q_s_a=", target_q_s_a)
        if weights != None:
            q_function_loss = self.q_function_criterion(q_s_a, target_q_s_a.detach()) * weights
        else:
            q_function_loss = self.q_function_criterion(q_s_a, target_q_s_a.detach())
            # q_function_loss = (q_s_a - target_q_s_a.detach()).pow(2).mean()
        # print("loss= ", q_function_loss)
        return q_function_loss

