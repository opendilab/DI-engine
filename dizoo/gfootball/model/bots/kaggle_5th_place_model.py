from kaggle_environments.envs.football.helpers import *
from math import sqrt
from enum import Enum
import torch
import torch.nn as nn
import numpy as np
from ding.torch_utils import tensor_to_list, one_hot, to_ndarray, to_tensor, to_dtype
from ding.utils import MODEL_REGISTRY
from .TamakEriFever.submission import agent


@MODEL_REGISTRY.register('football_kaggle_5th_place')
class FootballKaggle5thPlaceModel(torch.nn.Module):

    def __init__(self):
        super(FootballKaggle5thPlaceModel, self).__init__()
        # be compatiable with bc policy
        # to avoid: ValueError: optimizer got an empty parameter list
        self._dummy_param = nn.Parameter(torch.zeros(1, 1))

    def forward(self, data):
        actions = []
        data = data['raw_obs']
        if isinstance(data['score'], list):
            # to be compatiable with collect phase in subprocess mode
            data['score'] = torch.stack(data['score'], dim=-1)
        # dict of raw observations -> list of dict, each element in the list is the raw obs in a timestep
        data = [{k: v[i] for k, v in data.items()} for i in range(data['left_team'].shape[0])]
        for d in data:
            # the rew obs in one timestep
            if isinstance(d['steps_left'], torch.Tensor):
                d = {k: v.cpu() for k, v in d.items()}
                d = to_ndarray(d)
                for k in ['active', 'designated', 'ball_owned_player', 'ball_owned_team']:
                    d[k] = int(d[k])
                for k in ['sticky_actions']:
                    d[k] = list(d[k])
                d = {'controlled_players': 1, 'players_raw': [d]}
                actions.append(agent(d)[0])
        return {'action': torch.LongTensor(actions), 'logit': one_hot(torch.LongTensor(actions), 19)}
