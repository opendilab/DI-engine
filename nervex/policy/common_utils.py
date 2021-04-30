from typing import List, Dict, Any, Tuple, Union, Optional
from nervex.data import default_collate


def default_preprocess_learn(
        data: List[Any], use_priority: bool = False, use_nstep: bool = False, ignore_done: bool = False
) -> dict:
    # data preprocess
    data = default_collate(data)
    if ignore_done:
        data['done'] = None
    else:
        data['done'] = data['done'].float()
    if use_priority:
        data['weight'] = data['IS']
    else:
        data['weight'] = data.get('weight', None)
    if use_nstep:
        # Reward reshaping for n-step
        reward = data['reward']
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(1)
        # reward: (batch_size, nstep) -> (nstep, batch_size)
        data['reward'] = reward.permute(1, 0).contiguous()

    return data
