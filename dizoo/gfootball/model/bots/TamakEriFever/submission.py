import os.path as osp
import yaml

import numpy as np
import torch

from .football_ikki import Environment
from .handyrl_core.model import load_model

model_path = osp.join(osp.dirname(__file__), 'models/1679.pth')

with open(osp.join(osp.dirname(__file__), 'config.yaml')) as f:
    config = yaml.safe_load(f)

env = Environment(config['env_args'])
model = load_model(env.net()(env), model_path)
model.eval()


def output_think(env, obs, actions, p, v, r):
    pmask = np.ones_like(p)
    pmask[actions] = 0
    p = p - pmask * 1e32

    def softmax(x):
        x = np.exp(x - np.max(x, axis=-1))
        return x / x.sum(axis=-1)

    sticky_actions = obs['players_raw'][0]['sticky_actions']
    print(sticky_actions)

    print(actions)
    print((softmax(p) * 1000).astype(int))
    print(v)
    print(r)


prev_action = 0
reserved_action = None


def agent(obs):
    global prev_action, reserved_action

    info = [{'observation': obs, 'action': [prev_action]}, None]
    env.play_info(info)
    # print('step %d' % len(env.states))

    x = env.observation(0)

    p, v, r, _ = model.inference(x, None)
    actions = env.legal_actions(0)

    # output_think(env, obs, actions, p, v, r)

    ap_list = sorted([(a, p[a]) for a in actions], key=lambda x: -x[1])

    # you need return a list contains your single action(a int type number from [1, 18])
    # be ware of your model output might be a float number, so make sure return a int type number.
    action = ap_list[0][0]

    if reserved_action is not None:
        prev_action = reserved_action
        reserved_action = None
        # print('###RESERVED###')
    else:
        # split action
        prev_action, reserved_action = env.special_to_actions(action)

    return [prev_action]
