from typing import List
import numpy as np
from ding.utils import save_file

ID_COUNT = 0
np.random.seed(1)


def generate_data(meta: bool = False) -> dict:
    global ID_COUNT
    ret = {'obs': np.random.randn(4), 'data_id': str(ID_COUNT)}
    ID_COUNT += 1
    p_weight = np.random.uniform()
    if p_weight < 1 / 3:
        pass  # no key 'priority'
    elif p_weight < 2 / 3:
        ret['priority'] = None
    else:
        ret['priority'] = np.random.uniform() + 1e-3
    if not meta:
        return ret
    else:
        obs = ret.pop('obs')
        save_file(ret['data_id'], obs)
        return ret


def generate_data_list(count: int, meta: bool = False) -> List[dict]:
    return [generate_data(meta) for _ in range(0, count)]
