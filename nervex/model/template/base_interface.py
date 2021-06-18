import torch
import torch.nn as nn


class IActorCritic(nn.Module):
    r"""
    Overview:
        Abstract interface class for Actor-Critic based models
    Interface:
        forward, compute_actor_critic, compute_actor, compute_critic
    """

    def __init__(self) -> None:
        super(IActorCritic, self).__init__()
        self.modes = ['compute_actor_critic', 'compute_actor', 'compute_critic']

    def forward(self, inputs, mode: str, **kwargs):
        """
        Overview:
            Forward methods inherit from nn.Modules, used in different mode.
            Return the corresponding result according to the given mode arguments.
        Arguments:
            - inputs (:obj:`dict` or other :obj:`obj`): the input.
            - mode (:obj:`str`): the current mode to use in forward, support\
                ['compute_actor_critic', 'compute_actor', 'compute_critic']
        Returns:
            - return (:obj:`dict` or other :obj:`obj`): the correspond output.

        .. note::
            mode:
                - compute_actor_critic: normal reinforcement learning training
                - compute_actor: only use the actor part
                - compute_critic: only use the critic part
                - imitate: supervised learning, learn policy/value output label
        """
        assert (mode in self.modes)
        f = getattr(self, mode)
        return f(inputs, **kwargs)

    def compute_actor_critic(self, inputs):
        raise NotImplementedError

    def compute_actor(self, inputs, **kwargs):
        raise NotImplementedError

    def compute_critic(self, inputs, **kwargs):
        raise NotImplementedError
