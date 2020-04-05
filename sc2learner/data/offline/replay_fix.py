import torch
import copy
from sc2learner.envs import compress_obs, decompress_obs


def replay_fix(f):
    def fix(data, n):
        data = data.reshape(*data.shape[1:], data.shape[0])
        data = data.permute(2, 0, 1)
        return data
    replay = torch.load(f)
    dim_list = [2, 4, 2, 5, 2, 2, 2]
    new_data = []
    for idx, d in enumerate(replay):
        new_d = copy.deepcopy(decompress_obs(d))
        idx = 1
        for t in dim_list:
            new_d['spatial_info'][idx:idx+t] = fix(new_d['spatial_info'][idx:idx+t], t)
            idx = idx+t
        new_data.append(compress_obs(new_d))
    # output path should be modified
    torch.save(new_data, 'fix.step')