import torch
import torch.nn.functional as F


def fn(x):
    return x.unsqueeze(0).unsqueeze(0)


def tb_cross_entropy(logit, label):
    assert (len(label.shape) >= 2)
    T, B = label.shape[:2]
    # special 2D case
    if label.shape[2] == 2 and label.shape[2] != logit.shape[2]:
        assert (len(label.shape) == 3)
        n_output_shape = logit.shape[2:]
        label = label[..., 0] * n_output_shape[1] + label[..., 1]
        logit = logit.reshape(T, B, -1)

    label = label.reshape(-1)
    logit = logit.reshape(-1, logit.shape[-1])
    ce = F.cross_entropy(logit, label, reduction='none')
    ce = ce.reshape(T, B, -1)
    return ce.mean(dim=2)


def compute_importance_weights(
    target_output, behaviour_output, action, min_clip=None, max_clip=None, eps=1e-8, requires_grad=False, device='cpu'
):
    r"""
    Overview:
        Computing UPGO loss given constant gamma and lambda. There is no special handling for terminal state value.
        If zeros are passed in, output is not defined but will not be nans.
    Arguments:
        - target_output (:obj:`torch.Tensor`): the output computed by the target policy network,
          of size [T_traj, batchsize, n_output], n_output can be a list
        - behaviour_output (:obj:`torch.Tensor`):
          the output used producing the trajectory, of size [T_traj, batchsize, n_output]
        - action (:obj:`torch.Tensor`): the chosen action(index) in trajectory, of size [T_traj, batchsize] or
          [T_traj, batchsize, n_other]
        - min_clip (:obj:`float`): the lower bound of clip(default: None)
        - max_clip (:obj:`float`): the upper bound of clip(default: None)
        - eps (:obj:`float`): for numerical stability
    Returns:
        - rhos (:obj:`torch.Tensor`): Importance weight, of size [T_traj, batchsize]
    """
    grad_context = torch.enable_grad() if requires_grad else torch.no_grad()

    assert isinstance(action, torch.Tensor) or isinstance(action, list)

    with grad_context:
        if isinstance(action, list):
            T, B = len(action), len(action[0])
            rhos = torch.ones(T, B).to(device)
            for t in range(T):
                for b in range(B):
                    if action[t][b] is None:
                        rhos[t, b] = 1
                    else:
                        rhos[t, b] = tb_cross_entropy(fn(target_output[t][b]), fn(action[t][b])) / \
                                     (tb_cross_entropy(fn(behaviour_output[t][b]), fn(action[t][b])) + eps)
        else:
            rhos = tb_cross_entropy(target_output, action) / (tb_cross_entropy(behaviour_output, action) + eps)
            assert (rhos.shape == action.shape[:2])
        rhos = rhos.clamp(min_clip, max_clip)

        return rhos


def vtrace_advantages(clipped_rhos, clipped_cs, rewards, bootstrap_values, gammas=1.0, lambda_=0.8):
    r"""
    Overview:
        Computing vtrace advantages.
    Arguments:
        - clipped_rhos (:obj:`torch.Tensor`): clipped importance sampling weights $\rho$, of size [T_traj, batchsize]
        - clipped_cs (:obj:`torch.Tensor`): clipped importance sampling weights c, of size [T_traj, batchsize]
        - rewards (:obj:`torch.Tensor`): the returns from time step 0 to T-1, of size [T_traj, batchsize]
        - bootstrap_values (:obj:`torch.Tensor`): estimation of the state value at step 0 to T,
          of size [T_traj+1, batchsize]
    Returns:
        - result (:obj:`torch.Tensor`): Computed V-trace advantage, of size [T_traj, batchsize]
    """
    if not isinstance(gammas, torch.Tensor):
        gammas = gammas * torch.ones_like(rewards)
    if not isinstance(lambda_, torch.Tensor):
        lambda_ = lambda_ * torch.ones_like(rewards)
    deltas = clipped_rhos * (rewards + gammas * bootstrap_values[1:, :] - bootstrap_values[:-1, :])  # from 0 to T-1
    result = torch.empty_like(rewards)
    result[-1, :] = bootstrap_values[-2, :] + deltas[-1, :]
    for t in reversed(range(rewards.size()[0] - 1)):
        result[t, :] = bootstrap_values[t, :] + deltas[t, :] \
                       + gammas[t, :] * lambda_[t, :] * clipped_cs[t, :] * \
                       (result[t + 1, :] - bootstrap_values[t + 1, :])

    return result


def vtrace_loss(target_output, rhos, cs, action, rewards, bootstrap_values, gamma=1.0, lambda_=0.8):
    r"""
    Overview:
        Computing UPGO loss given constant gamma and lambda. There is no special handling for terminal state value,
        if the last state in trajectory is the terminal, just pass a 0 as bootstrap_terminal_value.
    Arguments:
        - target_output (:obj:`torch.Tensor`): the output computed by the target policy network,
          of size [T_traj, batchsize, n_output]
        - rhos (:obj:`torch.Tensor`): the clipped importance sampling ratio $\rho$, of size [T_traj, batchsize]
        - cs (:obj:`torch.Tensor`): the clipped importance sampling ratio c, of size [T_traj, batchsize]
        - action (:obj:`torch.Tensor`): the action taken, of size [T_traj, batchsize]
        - rewards (:obj:`torch.Tensor`): the returns from time step 0 to T-1, of size [T_traj, batchsize]
        - bootstrap_values (:obj:`torch.Tensor`): estimation of the state value at step 0 to T,
          of size [T_traj+1, batchsize]
    Returns:
        - loss (:obj:`torch.Tensor`): Computed V-trace loss, averaged over the samples, of size []
    """
    with torch.no_grad():
        advantages = vtrace_advantages(rhos, cs, rewards, bootstrap_values, gammas=gamma, lambda_=lambda_)
    if isinstance(action, list):
        T, B = len(action), len(action[0])
        metric = torch.zeros(T, B).to(dtype=rewards.dtype, device=rewards.device)
        for t in range(T):
            for b in range(B):
                if action[t][b] is None:
                    metric[t][b] = 0
                else:
                    metric[t][b] = tb_cross_entropy(fn(target_output[t][b]), fn(action[t][b]))
    else:
        metric = tb_cross_entropy(target_output, action)
        assert (metric.shape == action.shape[:2])
    losses = advantages * metric
    return -losses.mean()
