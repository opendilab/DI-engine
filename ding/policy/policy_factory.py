from typing import List, Dict, Any, Callable
from collections import namedtuple
import numpy as np
from numpy.core.overrides import ARRAY_FUNCTION_ENABLED

from ding.torch_utils import Adam, to_device
from ding.rl_utils import q_nstep_td_data, q_nstep_td_error
from ding.model import model_wrap
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn


class PolicyFactory:
    r"""
    Overview:
        Pure random policy. Only used for initial sample collecting if `cfg.policy.random_collect_size` > 0.
    """

    @staticmethod
    def get_random_policy(
            policy: 'BasePolicy',  # noqa
            action_space: 'EnvElementInfo' = None,  # noqa
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

            def discrete_random_action(min_val, max_val, shape):
                return np.random.randint(min_val, max_val, shape)

            def continuous_random_action(min_val, max_val, shape):
                bounded_below = min_val != float("inf")
                bounded_above = max_val != float("inf")
                unbounded = not bounded_below and not bounded_above
                low_bounded = bounded_below and not bounded_above
                upp_bounded = not bounded_below and bounded_above
                bounded = bounded_below and bounded_above
                assert sum([unbounded, low_bounded, upp_bounded, bounded]) == 1
                if unbounded:
                    return np.random.normal(size=unbounded[unbounded].shape)
                if low_bounded:
                    return np.random.exponential(size=shape) + min_val
                if upp_bounded:
                    return -np.random.exponential(size=shape) + max_val
                if bounded:
                    return np.random.uniform(low=min_val, high=max_val, size=shape)

            actions = {}
            discrete = action_space.value['dtype'] == int
            min, max, shape = action_space.value['min'], action_space.value['max'], action_space.shape
            for env_id in data:
                # For continuous env, action is limited in [-1, 1] for model output.
                # Env would scale it to its original action range.
                actions[env_id] = {
                    'action': discrete_random_action(min, max, shape)
                    if discrete else continuous_random_action(-1, 1, shape)
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
