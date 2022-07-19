from typing import Callable, Tuple, Union

from torch import Tensor
from ding.torch_utils import fold_batch, unfold_batch


def q_evaluation(obss: Tensor, actions: Tensor, q_critic_fn: Callable[[Tensor, Tensor],
                                                                      Tensor]) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Overview:
        Evaluate (observation, action) pairs along the trajectory

    Arguments:
        - obss (:obj:`torch.Tensor`): the observations along the trajectory
        - actions (:obj:`torch.Size`): the actions along the trajectory
        - q_critic_fn (:obj:`Callable`): the unified API :math:`Q(S_t, A_t)`

    Returns:
        - q_value (:obj:`torch.Tensor`): the action-value function evaluated along the trajectory

    Shapes:
        :math:`N`: time step
        :math:`B`: batch size
        :math:`O`: observation dimension
        :math:`A`: action dimension

        - obss:        [N, B, O]
        - actions:     [N, B, A]
        - q_value:     [N, B]

    """
    obss, dim = fold_batch(obss, 1)
    actions, _ = fold_batch(actions, 1)
    q_values = q_critic_fn(obss, actions)
    # twin critic
    if isinstance(q_values, list):
        return [unfold_batch(q_values[0], dim), unfold_batch(q_values[1], dim)]
    return unfold_batch(q_values, dim)
