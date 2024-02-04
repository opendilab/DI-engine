from typing import Dict, Any, Callable
from collections import namedtuple
from easydict import EasyDict
import gym
import gymnasium
import torch

from ding.torch_utils import to_device


class PolicyFactory:
    """
    Overview:
        Policy factory class, used to generate different policies for general purpose. Such as random action policy, \
        which is used for initial sample collecting for better exploration when ``random_collect_size`` > 0.
    Interfaces:
        ``get_random_policy``
    """

    @staticmethod
    def get_random_policy(
            policy: 'Policy.collect_mode',  # noqa
            action_space: 'gym.spaces.Space' = None,  # noqa
            forward_fn: Callable = None,
    ) -> 'Policy.collect_mode':  # noqa
        """
        Overview:
            According to the given action space, define the forward function of the random policy, then pack it with \
            other interfaces of the given policy, and return the final collect mode interfaces of policy.
        Arguments:
            - policy (:obj:`Policy.collect_mode`): The collect mode interfaces of the policy.
            - action_space (:obj:`gym.spaces.Space`): The action space of the environment, gym-style.
            - forward_fn (:obj:`Callable`): It action space is too complex, you can define your own forward function \
                and pass it to this function, note you should set ``action_space`` to ``None`` in this case.
        Returns:
            - random_policy (:obj:`Policy.collect_mode`): The collect mode intefaces of the random policy.
        """
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
                if isinstance(action_space, list):
                    if 'global_state' in data[env_id].keys():
                        # for smac
                        logit = torch.ones_like(data[env_id]['action_mask'])
                        logit[data[env_id]['action_mask'] == 0.0] = -1e8
                        dist = torch.distributions.categorical.Categorical(logits=torch.Tensor(logit))
                        actions[env_id] = {'action': dist.sample(), 'logit': torch.as_tensor(logit)}
                    else:
                        # for gfootball
                        actions[env_id] = {
                            'action': torch.as_tensor(
                                [action_space_agent.sample() for action_space_agent in action_space]
                            ),
                            'logit': torch.ones([len(action_space), action_space[0].n])
                        }
                elif isinstance(action_space, gymnasium.spaces.Dict):  # pettingzoo
                    actions[env_id] = {
                        'action': torch.as_tensor(
                            [action_space_agent.sample() for action_space_agent in action_space.values()]
                        )
                    }
                else:
                    if isinstance(action_space, gym.spaces.Discrete):
                        action = torch.LongTensor([action_space.sample()])
                    elif isinstance(action_space, gym.spaces.MultiDiscrete):
                        action = [torch.LongTensor([v]) for v in action_space.sample()]
                    else:
                        action = torch.as_tensor(action_space.sample())
                    actions[env_id] = {'action': action}
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


def get_random_policy(
        cfg: EasyDict,
        policy: 'Policy.collect_mode',  # noqa
        env: 'BaseEnvManager'  # noqa
) -> 'Policy.collect_mode':  # noqa
    """
    Overview:
        The entry function to get the corresponding random policy. If a policy needs special data items in a \
        transition, then return itself, otherwise, we will use ``PolicyFactory`` to return a general random policy.
    Arguments:
        - cfg (:obj:`EasyDict`): The EasyDict-type dict configuration.
        - policy (:obj:`Policy.collect_mode`): The collect mode interfaces of the policy.
        - env (:obj:`BaseEnvManager`): The env manager instance, which is used to get the action space for random \
            action generation.
    Returns:
        - random_policy (:obj:`Policy.collect_mode`): The collect mode intefaces of the random policy.
    """
    if cfg.policy.get('transition_with_policy_data', False):
        return policy
    else:
        action_space = env.action_space
        return PolicyFactory.get_random_policy(policy, action_space=action_space)
