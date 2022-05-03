from typing import Dict, Any, Callable
from collections import namedtuple
import torch
from ding.torch_utils import to_device
import numpy as np
import gym


class PolicyFactory:
    r"""
    Overview:
        Pure random policy. Only used for initial sample collecting if `cfg.policy.random_collect_size` > 0.
    """

    @staticmethod
    def get_random_policy(
            policy: 'BasePolicy',  # noqa
            action_space: 'gym.spaces.Space' = None,  # noqa
            forward_fn: Callable = None,
    ) -> None:
        assert not (action_space is None and forward_fn is None)
        random_collect_function = namedtuple(
            'random_collect_function', [
                'forward',
                'process_transition',
                'get_train_sample',
                'reset',
                'get_attribute',
            ]
        )

        def forward(data: Dict[int, Any], *args, **kwargs) -> Dict[int, Any]:

            actions = {}
            for env_id in data:
                if not isinstance(action_space, list):
                    action = action_space.sample()
                    if isinstance(action_space, gym.spaces.MultiDiscrete):
                        action = [torch.LongTensor([v]) for v in action]
                    actions[env_id] = {'action': action}
                elif 'global_state' in data[env_id].keys():
                    # for smac
                    logit = np.ones_like(data[env_id]['action_mask'])
                    logit[data[env_id]['action_mask'] == 0.0] = -1e8
                    dist = torch.distributions.categorical.Categorical(logits=torch.Tensor(logit))
                    actions[env_id] = {'action': np.array(dist.sample()), 'logit': np.array(logit)}
                else:
                    # for gfootball
                    actions[env_id] = {
                        'action': np.array([action_space_agent.sample() for action_space_agent in action_space]),
                        'logit': np.ones([len(action_space), action_space[0].n])
                    }

            return actions

        def reset(*args, **kwargs) -> None:
            pass

        if action_space is None:
            return random_collect_function(
                forward_fn, policy.process_transition, policy.get_train_sample, reset, policy.get_attribute
            )
        elif forward_fn is None:
            return random_collect_function(
                forward, policy.process_transition, policy.get_train_sample, reset, policy.get_attribute
            )
