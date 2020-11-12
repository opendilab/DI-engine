from collections import namedtuple
import torch

gae_data = namedtuple('gae_data', ['value', 'reward'])


def gae(data: namedtuple, gamma: float = 0.99, lambda_: float = 0.97) -> torch.FloatTensor:
    """
    Overview:
        Implementation of Generalized Advantage Estimator (arXiv:1506.02438)
    Arguments:
        - data (:obj:`namedtuple`): gae input data with fields ['value', 'reward'], which contains some episodes or\
        trajectories data
        - gamma (:obj:`float`): the future discount factor, should be in [0, 1], defaults to 0.99.
        - lambda (:obj:`float`): the gae parameter lambda, should be in [0, 1], defaults to 0.97, when lambda -> 0,\
        it induces bias, but when lambda -> 1, it has high variance due to the sum of terms.
    Returns:
        - adv (:obj:`torch.FloatTensor`): the calculated advantage
    Shapes:
        - value (:obj:`torch.FloatTensor`): :math:`(T+1, B)`, where T is trajectory length and B is batch size
        - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`
        - adv (:obj:`torch.FloatTensor`): :math:`(T, B)`
    .. note::
        value_{T+1} should be 0 if this trajectory reached a terminal state(done=True), otherwise we use value
        function, this operation is implemented in actor for packing trajectory.
    """
    value, reward = data
    delta = reward + gamma * value[1:] - value[:-1]
    factor = gamma * lambda_
    adv = torch.zeros_like(reward)
    gae_item = 0.
    for t in reversed(range(reward.shape[0])):
        gae_item = delta[t] + factor * gae_item
        adv[t] += gae_item
    return adv
