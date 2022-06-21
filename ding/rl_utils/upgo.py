import torch
import torch.nn.functional as F
from ding.hpc_rl import hpc_wrapper
from .td import generalized_lambda_returns


def tb_cross_entropy(logit, label, mask=None):
    assert (len(label.shape) >= 2)
    T, B = label.shape[:2]
    # Special 2D case
    if len(label.shape) > 2:
        assert len(label.shape) == 3
        s, n = logit.shape[-2:]
        logit = logit.reshape(-1, n)
        label = label.reshape(-1)
        ce = -F.cross_entropy(logit, label, reduction='none')
        ce = ce.view(T * B, -1)
        if mask is not None:
            ce *= mask.reshape(-1, s)
        ce = ce.sum(dim=1)
        ce = ce.reshape(T, B)
    else:
        label = label.reshape(-1)
        logit = logit.reshape(-1, logit.shape[-1])
        ce = -F.cross_entropy(logit, label, reduction='none')
        ce = ce.reshape(T, B, -1)
        ce = ce.mean(dim=2)
    return ce


def upgo_returns(rewards: torch.Tensor, bootstrap_values: torch.Tensor) -> torch.Tensor:
    r"""
    Overview:
        Computing UPGO return targets. Also notice there is no special handling for the terminal state.
    Arguments:
        - rewards (:obj:`torch.Tensor`): the returns from time step 0 to T-1, \
            of size [T_traj, batchsize]
        - bootstrap_values (:obj:`torch.Tensor`): estimation of the state value at step 0 to T, \
            of size [T_traj+1, batchsize]
    Returns:
        - ret (:obj:`torch.Tensor`): Computed lambda return value for each state from 0 to T-1, \
            of size [T_traj, batchsize]
    """
    # UPGO can be viewed as a lambda return! The trace continues for V_t (i.e. lambda = 1.0) if r_tp1 + V_tp2 > V_tp1.
    # as the lambdas[-1, :] is ignored in generalized_lambda_returns, we don't care about bootstrap_values_tp2[-1]
    lambdas = (rewards + bootstrap_values[1:]) >= bootstrap_values[:-1]
    lambdas = torch.cat([lambdas[1:], torch.ones_like(lambdas[-1:])], dim=0)
    return generalized_lambda_returns(bootstrap_values, rewards, 1.0, lambdas)


@hpc_wrapper(
    shape_fn=lambda args: args[0].shape,
    namedtuple_data=True,
    include_args=5,
    include_kwargs=['target_output', 'rhos', 'action', 'rewards', 'bootstrap_values']
)
def upgo_loss(
        target_output: torch.Tensor,
        rhos: torch.Tensor,
        action: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap_values: torch.Tensor,
        mask=None
) -> torch.Tensor:
    r"""
    Overview:
        Computing UPGO loss given constant gamma and lambda. There is no special handling for terminal state value,
        if the last state in trajectory is the terminal, just pass a 0 as bootstrap_terminal_value.
    Arguments:
        - target_output (:obj:`torch.Tensor`): the output computed by the target policy network, \
            of size [T_traj, batchsize, n_output]
        - rhos (:obj:`torch.Tensor`): the importance sampling ratio, of size [T_traj, batchsize]
        - action (:obj:`torch.Tensor`): the action taken, of size [T_traj, batchsize]
        - rewards (:obj:`torch.Tensor`): the returns from time step 0 to T-1, of size [T_traj, batchsize]
        - bootstrap_values (:obj:`torch.Tensor`): estimation of the state value at step 0 to T, \
            of size [T_traj+1, batchsize]
    Returns:
        - loss (:obj:`torch.Tensor`): Computed importance sampled UPGO loss, averaged over the samples, of size []
    """
    # discard the value at T as it should be considered in the next slice
    with torch.no_grad():
        returns = upgo_returns(rewards, bootstrap_values)
        advantages = rhos * (returns - bootstrap_values[:-1])
    metric = tb_cross_entropy(target_output, action, mask)
    assert (metric.shape == action.shape[:2])
    losses = advantages * metric
    return -losses.mean()
