import torch
import torch.nn as nn

class ActorCriticBase(nn.Module):
    r"""
    Overview:
        Abstract class for Actor-Critic based models
    Interface:
        forward, seed, compute_actor_critic, compute_actor, compute_critic, mimic, mimic_single
    """
    def __init__(self, ) -> None:
        self.modes = ['compute_actor_critic', 'compute_actor', 'compute_critic']

    def forward(self, inputs, mode=None, **kwargs):
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
                - mimic: supervised learning, learn policy/value output label
        """
        assert (mode in self.modes)
        f = getattr(self, mode)
        return f(inputs, **kwargs)

    def seed(self, seed):
        """
        Overview:
            Set the seed used in torch, see torch.manual_seed.
        """
        torch.manual_seed(seed)

    def compute_actor_critic(self, inputs):
        raise NotImplementedError

    def compute_actor(self, inputs, **kwargs):
        raise NotImplementedError

    def compute_critic(self, inputs, **kwargs):
        raise NotImplementedError

    def evaluate(self, inputs, **kwargs):
        raise NotImplementedError

    def _actor_forward(self, inputs, **kwargs):
        raise NotImplementedError

    def _critic_forward(self, inputs, **kwargs):
        raise NotImplementedError

class ValueActorCriticBase(nn.Module):
    r"""
    Overview:
        Abstract class for Value based Actor-Critic based models
    Interface:
        forward, seed, compute_action_value, compute_action, mimic
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


class QActorCriticBase(nn.Module):
    r"""
    Overview:
        Abstract class for Q_value based Actor-Critic based models
    Interface:
        forward, seed, optimize_actor, compute_q, compute_action, mimic
    """

    def forward(self, inputs, mode=None, **kwargs):
        """
        Overview:
            Forward methods inherit from nn.Modules, used in different mode.
            Return the corresponding result according to the given mode arguments.
        Arguments:
            - inputs (:obj:`dict` or other :obj:`obj`): the input.
            - mode (:obj:`str`): the current mode to use in forward, support\
                ['compute_q, 'compute_action', 'optimize_acto', 'mimic']
        Returns:
            - return (:obj:`dict` or other :obj:`obj`): the correspond output.

        .. note::
            mode:
                - optimize_actor: optimize actor part, with critic part `no grad`, return q value
                - compute_q: evaluate q value based on state and action from buffer
                - compute_action: evaluate policy performance, only use the actor part
                - mimic: supervised learning, learn policy/value output label
        """
        assert (mode in ['optimize_actor', 'compute_q', 'compute_action', 'compute_action_q', 'mimic']), mode
        f = getattr(self, mode)
        return f(inputs, **kwargs)

    def seed(self, seed):
        torch.manual_seed(seed)

    def optimize_actor(self, inputs, **kwargs):
        raise NotImplementedError

    def compute_q(self, inputs, **kwargs):
        raise NotImplementedError

    def compute_action(self, inputs, **kwargs):
        raise NotImplementedError

    def mimic(self, inputs, **kwargs):
        raise NotImplementedError

    def _actor_forward(self, inputs, **kwargs):
        raise NotImplementedError

    def _critic_forward(self, inputs, **kwargs):
        raise NotImplementedError


class SoftActorCriticBase(nn.Module):

    def forward(self, inputs, mode=None, **kwargs):
        """
        Note:
            mode:
                - evaluate: use re-parameterization tick
                - compute_q: compute q value using the soft q network
                - compute_value: compute value using the value network
                - compute_action: compute action useing the policy network
                - mimic: supervised learning, learn policy/value output label
        """
        assert (mode in ['evaluate', 'compute_value', 'compute_q', 'compute_action', 'mimic'])
        f = getattr(self, mode)
        return f(inputs, **kwargs)

    def seed(self, seed):
        torch.manual_seed(seed)

    def evaluate(self, inputs):
        raise NotImplementedError

    def compute_action(self, inputs, **kwargs):
        raise NotImplementedError

    def compute_q(self, inputs, **kwargs):
        raise NotImplementedError

    def compute_value(self, inputs, **kwargs):
        raise NotImplementedError

    def mimic(self, inputs, **kwargs):
        raise NotImplementedError

    def _value_net_forward(self, inputs, **kwargs):
        raise NotImplementedError

    def _soft_q_net_forward(self, inputs, **kwargs):
        raise NotImplementedError

    def _policy_net_forward(self, inputs, **kwargs):
        raise NotImplementedError


class PhasicPolicyGradientBase(nn.Module):

    def forward(self, inputs, mode=None, **kwargs):
        """
        Note:
            mode:
                - compute_action_value: compute action using the policy network \
                and compute value using the value network
                - compute_action: compute action using the policy network
                - compute_value: compute value using the value network
                - compute_policy_value: compute value using the auxiliary value head \
                in policy network
                - mimic: supervised learning, learn policy/value output label
        """
        assert (mode in ['compute_action_value', 'compute_action', 'compute_value'])
        f = getattr(self, mode)
        return f(inputs, **kwargs)

    def seed(self, seed):
        torch.manual_seed(seed)

    def compute_action_value(self, inputs, **kwargs):
        raise NotImplementedError

    def compute_action(self, inputs, **kwargs):
        raise NotImplementedError

    def compute_value(self, inputs, **kwargs):
        raise NotImplementedError

    def _value_net_forward(self, inputs, **kwargs):
        raise NotImplementedError

    def _policy_net_forward(self, inputs, **kwargs):
        raise NotImplementedError
