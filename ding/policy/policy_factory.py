from typing import Dict, Any, Callable
from collections import namedtuple

from ding.torch_utils import to_device


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
                actions[env_id] = {'action': action_space.sample()}
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
