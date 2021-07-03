from kaggle_environments.envs.football.helpers import *
from math import sqrt
from enum import Enum
import torch
import numpy as np
from ding.torch_utils import tensor_to_list, one_hot
from .TamakEriFever.submission import agent


class FootballKaggle5thPlaceModel(torch.nn.Module):

    def __init__(self):
        super(FootballKaggle5thPlaceModel, self).__init__()

    def forward(self, data):
        actions = []
        for d in data:
            if isinstance(d['steps_left'], torch.Tensor):
                for k, v in d.items():
                    v = tensor_to_list(v)
                    if len(v) == 1:
                        v = int(v[0])
                    # else:
                    #     v = np.array(v)
                    d[k] = v
                d = {'controlled_players': 1, 'players_raw': [d]}
                # print("current d = ", d)
                actions.append(agent(d)[0])
        return {'action': torch.LongTensor(actions), 'logit': one_hot(torch.LongTensor(actions), 19)}
