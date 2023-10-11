from typing import Union
import torch
from torch.distributions import Categorical, Independent, Normal


def compute_importance_weights(
    target_output: Union[torch.Tensor, dict],
    behaviour_output: Union[torch.Tensor, dict],
    action: torch.Tensor,
    action_space_type: str = 'discrete',
    requires_grad: bool = False
):
    """
    Overview:
        Computing importance sampling weight with given output and action
    Arguments:
        - target_output (:obj:`Union[torch.Tensor,dict]`): the output taking the action \
            by the current policy network, \
            usually this output is network output logit if action space is discrete, \
            or is a dict containing parameters of action distribution if action space is continuous.
        - behaviour_output (:obj:`Union[torch.Tensor,dict]`): the output taking the action \
            by the behaviour policy network,\
            usually this output is network output logit,  if action space is discrete, \
            or is a dict containing parameters of action distribution if action space is continuous.
        - action (:obj:`torch.Tensor`): the chosen action(index for the discrete action space) in trajectory,\
            i.e.: behaviour_action
        - action_space_type (:obj:`str`): action space types in ['discrete', 'continuous']
        - requires_grad (:obj:`bool`): whether requires grad computation
    Returns:
        - rhos (:obj:`torch.Tensor`): Importance sampling weight
    Shapes:
        - target_output (:obj:`Union[torch.FloatTensor,dict]`): :math:`(T, B, N)`, \
            where T is timestep, B is batch size and N is action dim
        - behaviour_output (:obj:`Union[torch.FloatTensor,dict]`): :math:`(T, B, N)`
        - action (:obj:`torch.LongTensor`): :math:`(T, B)`
        - rhos (:obj:`torch.FloatTensor`): :math:`(T, B)`
    Examples:
        >>> target_output = torch.randn(2, 3, 4)
        >>> behaviour_output = torch.randn(2, 3, 4)
        >>> action = torch.randint(0, 4, (2, 3))
        >>> rhos = compute_importance_weights(target_output, behaviour_output, action)
    """
    grad_context = torch.enable_grad() if requires_grad else torch.no_grad()
    assert isinstance(action, torch.Tensor)
    assert action_space_type in ['discrete', 'continuous']

    with grad_context:
        if action_space_type == 'continuous':
            dist_target = Independent(Normal(loc=target_output['mu'], scale=target_output['sigma']), 1)
            dist_behaviour = Independent(Normal(loc=behaviour_output['mu'], scale=behaviour_output['sigma']), 1)
            rhos = dist_target.log_prob(action) - dist_behaviour.log_prob(action)
            rhos = torch.exp(rhos)
            return rhos
        elif action_space_type == 'discrete':
            dist_target = Categorical(logits=target_output)
            dist_behaviour = Categorical(logits=behaviour_output)
            rhos = dist_target.log_prob(action) - dist_behaviour.log_prob(action)
            rhos = torch.exp(rhos)
            return rhos
