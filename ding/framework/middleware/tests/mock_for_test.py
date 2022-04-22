
from typing import Union, Any, List, Callable, Dict, Optional
from collections import namedtuple
import torch
import treetensor.numpy as tnp
from easydict import EasyDict
from unittest.mock import Mock

obs_dim = [2, 2]
action_space = 1
env_num = 2


CONFIG = dict(
    seed=0,
    policy=dict(
        learn=dict(
            learner=dict(
                hook=dict(log_show_after_iter=10),
            ),
        ),
        eval=dict(
            evaluator=dict(eval_freq=100),
        ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
        ),
    ),
    env=dict(
        n_evaluator_episode=4,
        stop_value=0.9,
    ),
    ctx=dict(
        collect_kwargs=dict(eps=0.1),
        last_eval_iter=-1,
        train_iter=0,
        env_step=0,
        env_episode=0,
    ),
)
CONFIG = EasyDict(CONFIG)


class MockPolicy(Mock):
    def __init__(self) -> None:
        super(MockPolicy, self).__init__()
        self.action_space = action_space
        self.obs_dim = obs_dim

    def reset(self, data_id: Optional[List[int]] = None) -> None:
        return

    def forward(self, data: dict, **kwargs) -> dict:
        res = {}
        for i, v in data.items():
            res[i] = {'action': torch.sum(v)}
        return res
    
    def process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        transition = {
            'obs': torch.rand(self.obs_dim),
            'next_obs': torch.rand(self.obs_dim),
            'action': torch.zeros(self.action_space),
            'logit': 1.0,
            'value': 2.0,
            'reward': 0.1,
            'done': True,
        }
        return transition

class MockEnv(Mock):
    def __init__(self) -> None:
        super(MockEnv, self).__init__()
        self.env_num = env_num
        self.obs_dim = obs_dim
        self.closed = False
    
    @property
    def ready_obs(self) -> tnp.array:
        return tnp.stack([
            torch.zeros(self.obs_dim),
            torch.ones(self.obs_dim),
        ])

    def seed(self, seed: Union[Dict[int, int], List[int], int], dynamic_seed: bool = None) -> None:
        return

    def launch(self, reset_param: Optional[Dict] = None) -> None:
        return
    
    def reset(self, reset_param: Optional[Dict] = None) -> None:
        return

    def step(self, actions: tnp.ndarray) -> List[tnp.ndarray]:
        timesteps = []
        for i in range(self.env_num):
            timestep = dict()
            # TODO:
            timestep['obs'] = torch.rand(self.obs_dim)
            timestep['reward'] = 1.0
            timestep['done'] = True
            timestep['info'] = {'final_eval_reward': 1.0}
            timestep['env_id'] = i

            timesteps.append(tnp.array(timestep))
        return timesteps