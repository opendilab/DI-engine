import torch
from torch.distributions import Categorical, Normal, Independent


def compute_importance_weights(target_output, behaviour_output, action, action_space='discrete', requires_grad=False):
    """
    Overview:
        Computing importance sampling weight with given output and action
    Arguments:
        - target_output (:obj:`torch.Tensor`): the output taking the action by the current policy network,\
            usually this output is network output logit
        - behaviour_output (:obj:`torch.Tensor`): the output taking the action by the behaviour policy network,\
            usually this output is network output logit, which is used to produce the trajectory(collector)
        - action (:obj:`torch.Tensor`): the chosen action(index for the discrete action space) in trajectory,\
            i.e.: behaviour_action
        - requires_grad (:obj:`bool`): whether requires grad computation
    Returns:
        - rhos (:obj:`torch.Tensor`): Importance sampling weight
    Shapes:
        - target_output (:obj:`torch.FloatTensor`): :math:`(T, B, N)`, where T is timestep, B is batch size and\
            N is action dim
        - behaviour_output (:obj:`torch.FloatTensor`): :math:`(T, B, N)`
        - action (:obj:`torch.LongTensor`): :math:`(T, B)`
        - rhos (:obj:`torch.FloatTensor`): :math:`(T, B)`
    """
    assert isinstance(action, torch.Tensor)

    with torch.set_grad_enabled(requires_grad):
        if action_space == 'discrete':
            dist_target = Categorical(logits=target_output)
            dist_behaviour = Categorical(logits=behaviour_output)
        else:
            # mu, sigma = target_output
            dist_target = Independent(Normal(*target_output), 1)
            dist_behaviour = Independent(Normal(*behaviour_output), 1)
        rhos = dist_target.log_prob(action) - dist_behaviour.log_prob(action)
        rhos = torch.exp(rhos)
        return rhos
