from typing import Dict, Any, Callable
from collections import namedtuple
from easydict import EasyDict
import torch

from ding.torch_utils import to_device
import numpy as np


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
<<<<<<< HEAD
                if not isinstance(action_space, list):
                    actions[env_id] = {'action': action_space.sample()}
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

=======
                actions[env_id] = {'action': torch.as_tensor(action_space.sample())}
>>>>>>> bcfc76f255f0be1e8edbecb8dee3ac1438964c77
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


def get_random_policy(cfg: EasyDict, policy: 'Policy.collect_mode', env: 'BaseEnvManager'):  # noqa
    if cfg.policy.get('transition_with_policy_data', False):
        return policy
    else:
        action_space = env.action_space
        return PolicyFactory.get_random_policy(policy, action_space=action_space)
