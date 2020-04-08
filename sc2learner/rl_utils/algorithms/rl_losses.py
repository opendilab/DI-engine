"""Library for RL returns and losses evaluation"""

import math
from functools import reduce
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


def multistep_forward_view(rewards, gammas, bootstrap_values, lambda_):
    r"""
    Overview:
        Same as trfl.sequence_ops.multistep_forward_view
        Implementing (12.18) in Sutton & Barto
        ```
        result[T-1] = rewards[T-1] + gammas[T-1] * bootstrap_values[T]
        for t in 0...T-2 :
        result[t] = rewards[t] + gammas[t]*(lambdas[t]*result[t+1] + (1-lambdas[t])*bootstrap_values[t+1])
        ```
        Assuming the first dim of input tensors correspond to the index in batch
        There is no special handling for terminal state value,
        if some state has reached the terminal, just fill in zeros for values and rewards beyond terminal
        (including the terminal state, which is, bootstrap_values[terminal] should also be 0)
    Arguments:
        - rewards (:obj:`torch.Tensor`): the returns from 0 to T-1, of size [T_traj, batchsize]
        - gammas (:obj:`torch.Tensor`): discount factor for each step (from 0 to T-1), of size [T_traj, batchsize]
        - bootstrap_values (:obj:`torch.Tensor`): estimation of the value at *step 1 to T*, of size [T_traj, batchsize]
        - lambda_ (:obj:`torch.Tensor`): determining the mix of bootstrapping
        vs further accumulation of multistep returns at each timestep of size [T_traj, batchsize],
        the element for T-1 is ignored and effectively set to 0,
        as there is no information about future rewards.
    Returns:
        - ret (:obj:`torch.Tensor`): Computed lambda return value
         for each state from 0 to T-1, of size [T_traj, batchsize]
    """
    result = torch.empty_like(rewards)
    # Forced cutoff at the last one
    result[-1, :] = rewards[-1, :] + gammas[-1, :] * bootstrap_values[-1, :]
    discounts = gammas * lambda_
    for t in reversed(range(rewards.size()[0] - 1)):
        result[t, :] = rewards[t, :]\
            + discounts[t, :] * result[t+1, :]\
            + (gammas[t, :] - discounts[t, :]) * bootstrap_values[t, :]

    return result


def generalized_lambda_returns(rewards, gammas, bootstrap_values, lambda_):
    r"""
    Overview:
        Functional equivalent to trfl.value_ops.generalized_lambda_returns
        https://github.com/deepmind/trfl/blob/2c07ac22512a16715cc759f0072be43a5d12ae45/trfl/value_ops.py#L74
        Passing in a number instead of tensor to make the value constant for all samples in batch
    Arguments:
        - rewards (:obj:`torch.Tensor`): the returns from 0 to T-1, of size [T_traj, batchsize]
        - gammas (:obj:`torch.Tensor` or :obj:`float`):
          discount factor for each step (from 0 to T-1), of size [T_traj, batchsize]
        - bootstrap_values (:obj:`torch.Tensor` or :obj:`float`):
          estimation of the value at step 0 to *T*, of size [T_traj+1, batchsize]
        - lambda_ (:obj:`torch.Tensor` or :obj:`float`): determining the mix of bootstrapping
          vs further accumulation of multistep returns at each timestep, of size [T_traj, batchsize]
    Returns:
        - ret (:obj:`torch.Tensor`): Computed lambda return value
          for each state from 0 to T-1, of size [T_traj, batchsize]
    """
    if not isinstance(gammas, torch.Tensor):
        gammas = gammas * torch.ones_like(rewards)
    if not isinstance(lambda_, torch.Tensor):
        lambda_ = lambda_ * torch.ones_like(rewards)
    bootstrap_values_tp1 = bootstrap_values[1:, :]
    return multistep_forward_view(rewards, gammas, bootstrap_values_tp1, lambda_)


def td_lambda_loss(values, rewards, gamma=1.0, lambda_=0.8):
    r"""
    Overview:
        Computing TD($\lambda$) loss given constant gamma and lambda.
        There is no special handling for terminal state value,
        if some state has reached the terminal, just fill in zeros for values and rewards beyond terminal
        (*including the terminal state*, values[terminal] should also be 0)
    Arguments:
        - values (:obj:`torch.Tensor`): estimation of the state value at step 0 to T, of size [T_traj+1, batchsize]
        - rewards (:obj:`torch.Tensor`): the returns from time step 0 to T-1, of size [T_traj, batchsize]
        - gamma (:obj:`float`): constant gamma
        - lambda_ (:obj:`float`): constant lambda (between 0 to 1)
    Returns:
        - loss (:obj:`torch.Tensor`): Computed MSE loss, averaged over the batch, of size []
    """
    with torch.no_grad():
        returns = generalized_lambda_returns(rewards, gamma, values, lambda_)
    # discard the value at T as it should be considered in the next slice
    loss = 0.5 * torch.pow(returns - values[:-1], 2).mean()
    return loss


def compute_importance_weights(
    target_output,
    behaviour_output,
    output_type,
    action,
    min_clip=None,
    max_clip=None,
    eps=1e-8,
    requires_grad=False,
    device='cpu'
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
        - output_type (:obj:`str`): the type of target/behaviour output(value, logit)
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
        if output_type == 'value':
            rhos = torch.clamp(target_output / (behaviour_output + eps), max=3)  # action_logits can be zero
            rhos = rhos.mean(dim=2)
        elif output_type == 'logit':
            if isinstance(action, list):
                T, B = len(action), len(action[0])
                rhos = torch.ones(T, B).to(device)
                for t in range(T):
                    for b in range(B):
                        if action[t][b] is None:
                            rhos[t, b] = 1
                        else:
                            rhos[t, b] = tb_cross_entropy(fn(target_output[t][b]), fn(action[t][b])) /\
                                         (tb_cross_entropy(fn(behaviour_output[t][b]), fn(action[t][b])) + eps)
            else:
                rhos = tb_cross_entropy(target_output, action) / (tb_cross_entropy(behaviour_output, action) + eps)
                assert (rhos.shape == action.shape[:2])
        else:
            raise RuntimeError("not support target output type: {}".format(output_type))
        rhos = rhos.clamp(min_clip, max_clip)

        return rhos


def upgo_returns(rewards, bootstrap_values):
    r"""
    Overview:
        Computing UPGO return targets. Also notice there is no special handling for the terminal state.
    Arguments:
        - rewards (:obj:`torch.Tensor`): the returns from time step 0 to T-1, of size [T_traj, batchsize]
        - bootstrap_values (:obj:`torch.Tensor`):
          estimation of the state value at step 0 to T, of size [T_traj+1, batchsize]
    Returns:
        - ret (:obj:`torch.Tensor`): Computed lambda return value
          for each state from 0 to T-1, of size [T_traj, batchsize]
    """
    # UPGO can be viewed as a lambda return! The trace continues for V_t (i.e. lambda = 1.0) if r_tp1 + V_tp2 > V_tp1.
    # as the lambdas[-1, :] is ignored in generalized_lambda_returns, we don't care about bootstrap_values_tp2[-1]
    bootstrap_values_tp1 = bootstrap_values[1:, :]
    bootstrap_values_tp2 = torch.cat((bootstrap_values_tp1[1:, :], bootstrap_values[-1, :].unsqueeze(0)), 0)
    lambdas = 1.0 * (rewards + bootstrap_values_tp2) >= bootstrap_values_tp1
    return generalized_lambda_returns(rewards, 1.0, bootstrap_values, lambdas)


def upgo_loss(target_output, output_type, rhos, action, rewards, bootstrap_values):
    r"""
    Overview:
        Computing UPGO loss given constant gamma and lambda. There is no special handling for terminal state value,
        if the last state in trajectory is the terminal, just pass a 0 as bootstrap_terminal_value.
    Arguments:
        - target_output (:obj:`torch.Tensor`): the output computed by the target policy network,
          of size [T_traj, batchsize, n_output]
        - output_type (:obj:`str`): the type of target output(value, logit)
        - rhos (:obj:`torch.Tensor`): the importance sampling ratio, of size [T_traj, batchsize]
        - action (:obj:`torch.Tensor`): the action taken, of size [T_traj, batchsize]
        - rewards (:obj:`torch.Tensor`): the returns from time step 0 to T-1, of size [T_traj, batchsize]
        - bootstrap_values (:obj:`torch.Tensor`): estimation of the state value at step 0 to T,
          of size [T_traj+1, batchsize]
    Returns:
        - loss (:obj:`torch.Tensor`): Computed importance sampled UPGO loss, averaged over the samples, of size []
    """
    # discard the value at T as it should be considered in the next slice
    with torch.no_grad():
        returns = upgo_returns(rewards, bootstrap_values)
        advantages = rhos * (returns - bootstrap_values[:-1])
    if output_type == 'value':
        metric = F.l1_loss(target_output, action.float(), reduction='none')
        metric = metric.mean(dim=2)
    elif output_type == 'logit':
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
    else:
        raise RuntimeError("not support target output type: {}".format(output_type))
    losses = advantages * metric
    return -losses.mean()


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
    deltas = clipped_rhos * \
        (rewards + gammas *
         bootstrap_values[1:, :] - bootstrap_values[:-1, :])  # from 0 to T-1
    result = torch.empty_like(rewards)
    result[-1, :] = bootstrap_values[-2, :] + deltas[-1, :]
    for t in reversed(range(rewards.size()[0] - 1)):
        result[t, :] = bootstrap_values[t, :] + deltas[t, :]\
            + gammas[t, :] * lambda_[t, :] * clipped_cs[t, :] * \
            (result[t+1, :] - bootstrap_values[t+1, :])

    return result


def vtrace_loss(target_output, output_type, rhos, cs, action, rewards, bootstrap_values, gamma=1.0, lambda_=0.8):
    r"""
    Overview:
        Computing UPGO loss given constant gamma and lambda. There is no special handling for terminal state value,
        if the last state in trajectory is the terminal, just pass a 0 as bootstrap_terminal_value.
    Arguments:
        - target_output (:obj:`torch.Tensor`): the output computed by the target policy network,
          of size [T_traj, batchsize, n_output]
        - output_type (:obj:`str`): the type of target output(value, logit)
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
    if output_type == 'value':
        metric = F.l1_loss(target_output, action.float(), reduction='none')
        metric = metric.mean(dim=2)
    elif output_type == 'logit':
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
    else:
        raise RuntimeError("not support target output type: {}".format(output_type))
    losses = advantages * metric
    return -losses.mean()


def entropy(policy_logits, masked_threshold=-1e3):
    r"""
    Overview:
        Computing entropy given the logits
    Arguments:
        - policy_logits (:obj:`torch.Tensor`): the logits computed by policy network,
          of size [T_traj, batchsize, n_action_type]
    Returns:
        - entropy (:obj:`torch.Tensor`): Computed entropy, averaged over the samples, of size []
    """
    # mask all the masked logits in entropy computation
    valid_flag = torch.where(
        policy_logits > masked_threshold, torch.ones_like(policy_logits), torch.zeros_like(policy_logits)
    )
    entropy = -F.softmax(policy_logits, dim=-1) * F.log_softmax(policy_logits, dim=-1)
    entropy = entropy * valid_flag
    entropy = torch.mean(entropy)
    # Normalize by actions available.
    numel = reduce(lambda x, y: x * y, policy_logits.shape)
    entropy = entropy * numel / torch.sum(valid_flag)
    return entropy


def pg_loss(
    action_logits,
    current_logits,
    action,
    rewards,
    bootstrap_values,
    clipping=lambda x: torch.clamp(x, min=1),
    upgo_weight=0.5,
    ent_weight=0.2,
    vtrace_gamma=1.0,
    vtrace_lambda=0.8
):
    r"""
    Overview:
        Computing total policy gradient loss.
        There is no special handling for terminal state value, just fill in zeros as input for terminated game.
    Arguments:
        - action_logits (:obj:`torch.Tensor`): the logits used producing the trajectory,
          of size [T_traj, batchsize, n_action_type]
        - current_logits (:obj:`torch.Tensor`): the logits computed by the target policy network,
          of size [T_traj, batchsize, n_action_type]
        - action (:obj:`torch.Tensor`): the index of action taken (in range(0,n_action_type)),
          of size [T_traj, batchsize]
        - rewards (:obj:`torch.Tensor`): the returns from time step 0 to T-1, of size [T_traj, batchsize]
        - bootstrap_values (:obj:`torch.Tensor`): estimation of the state value at step 0 to T,
          of size [T_traj+1, batchsize]
        - clipping: the clipping function for the importance weight,
          the raw IW tensor of size [T_traj, batchsize] is passed in, should return clipped value
        - upgo_weight (:obj:`float`): weight for upgo
        - ent_weight (:obj:`float`): weight for entropy
        - vtrace_gamma
        - vtrace_lambda
    Returns:
        - loss (:obj:`torch.Tensor`): Computed total loss, averaged over the samples, of size []
    """
    rhos = compute_importance_weights(action_logits, current_logits, action, clipping=clipping, need_grad=False)
    cs = rhos
    return vtrace_loss(current_logits, rhos, cs, action, rewards, bootstrap_values,
                       gamma=vtrace_gamma, lambda_=vtrace_lambda)\
        + upgo_weight * upgo_loss(current_logits, rhos, action, rewards, bootstrap_values)\
        - ent_weight * entropy(current_logits)
