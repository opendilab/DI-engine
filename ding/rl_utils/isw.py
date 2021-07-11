import torch


def compute_importance_weights(target_output, behaviour_output, action, requires_grad=False):
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
    grad_context = torch.enable_grad() if requires_grad else torch.no_grad()
    assert isinstance(action, torch.Tensor)

    with grad_context:
        dist_target = torch.distributions.Categorical(logits=target_output)
        dist_behaviour = torch.distributions.Categorical(logits=behaviour_output)
        rhos = dist_target.log_prob(action) - dist_behaviour.log_prob(action)
        rhos = torch.exp(rhos)
        return rhos
