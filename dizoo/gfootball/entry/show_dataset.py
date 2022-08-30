"""
Overview:
    The following is to show some statistics of the dataset in gfootball env.
"""
import torch
import numpy as np
import os
from ding.config import read_config, compile_config
from ding.utils.data import create_dataset
from dizoo.gfootball.entry.gfootball_bc_config import main_config, create_config

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

if __name__ == "__main__":
    config = [main_config, create_config]
    input_cfg = config
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    cfg = compile_config(cfg, seed=0, auto=True, create_cfg=create_cfg)
    cfg.policy.collect.data_type = 'naive'
    """episode data"""
    # Users should add their own BC data path here.
    cfg.policy.collect.data_path = dir_path + '/gfootball_rule_100eps.pkl'
    dataset = create_dataset(cfg)

    print('num_episodes', dataset.__len__())
    print('episode 0, transition 0', dataset.__getitem__(0)[0])
    episodes_len = np.array([len(dataset.__getitem__(i)) for i in range(dataset.__len__())])
    print('episodes_len', episodes_len)
    return_of_episode = torch.stack(
        [
            torch.stack(
                [dataset.__getitem__(episode)[i]['reward'] for i in range(dataset.__getitem__(episode).__len__())],
                axis=0
            ).sum(0) for episode in range(dataset.__len__())
        ],
        axis=0
    )
    print('return_of_episode', return_of_episode)
    print(return_of_episode.mean(), return_of_episode.max(), return_of_episode.min())
    """transition data"""
    # Users should add their own BC data path here.
    cfg.policy.collect.data_path = dir_path + '/gfootball_rule_100eps_transitions_lt0.pkl'
    dataset = create_dataset(cfg)

    print('num_transitions', dataset.__len__())
    print('transition 0: ', dataset.__getitem__(0))

    reward_of_transitions = torch.stack(
        [dataset.__getitem__(transition)['reward'] for transition in range(dataset.__len__())], axis=0
    )
    print('reward_of_transitions', reward_of_transitions)
    print(reward_of_transitions.mean(), reward_of_transitions.max(), reward_of_transitions.min())
