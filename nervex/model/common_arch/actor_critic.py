import torch
import torch.nn as nn


class ActorCriticBase(nn.Module):
    r"""
    Overview:
        Abstract class for Actor-Critic based models
    Interface:
        forward, set_seed, step, value, evaluate, mimic
    """

    def forward(self, inputs, mode=None, **kwargs):
        """
        Overview:
            Forward methods inherit from nn.Modules, used in different mode.
            Return the corresponding result according to the given mode arguments.
        Arguments:
            - inputs (:obj:`dict` or other :obj:`obj`): the input.
            - mode (:obj:`str`): the current mode to use in forward, support\
                ['compute_action_value', 'compute_action', 'mimic']
        Returns:
            - return (:obj:`dict` or other :obj:`obj`): the correspond output.

        .. note::
            mode:
                - compute_action_value: normal reinforcement learning training
                - compute_action: evaluate policy performance, only use the actor part
                - mimic: supervised learning, learn policy/value output label
        """
        assert (mode in ['compute_action_value', 'compute_action', 'mimic', 'mimic_single'])
        f = getattr(self, mode)
        return f(inputs, **kwargs)

    def seed(self, seed):
        """
        Overview:
            Set the seed used in torch, see torch.manual_seed.
        """
        torch.manual_seed(seed)

    def compute_action_value(self, inputs):
        raise NotImplementedError

    def compute_action(self, inputs, **kwargs):
        raise NotImplementedError

    def mimic(self, inputs, **kwargs):
        raise NotImplementedError

    def _actor_forward(self, inputs, **kwargs):
        raise NotImplementedError

    def _critic_forward(self, inputs, **kwargs):
        raise NotImplementedError
