"""Library for RL returns and losses evaluation"""

import torch
import torch.nn.functional as F


def multistep_forward_view(rewards, gammas, bootstrap_values, lambda_, need_grad=False):
    r"""
    Overview:
        Same as trfl.sequence_ops.multistep_forward_view
        Implementing (12.18) in Sutton & Barto
        ```
        result[T-1] = rewards[T-1] + gammas[T-1] * bootstrap_values[T]
        for t in 0...T-2 :
        result[t] = rewards[t] + gammas[t]*(lambdas[t]*result[t+1] + (1-lambdas[t])*bootstrap_values[t+1])
        ```
        Assumming the first dim of input tensors correspond to the index in batch
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
    grad_mode = torch.is_grad_enabled()
    torch.set_grad_enabled(need_grad)

    result = torch.empty(rewards.size())
    # Forced cutoff at the last one
    result[-1, :] = rewards[-1, :] + gammas[-1, :] * bootstrap_values[-1, :]
    discounts = gammas * lambda_
    for t in reversed(range(rewards.size()[0] - 1)):
        result[t, :] = rewards[t, :]\
            + discounts[t, :] * result[t+1, :]\
            + (gammas[t, :] - discounts[t, :]) * bootstrap_values[t, :]

    torch.set_grad_enabled(grad_mode)
    return result


def generalized_lambda_returns(rewards, gammas, bootstrap_values, lambda_, need_grad=False):
    r"""
    Overview:
        Functional equvalent to trfl.value_ops.generalized_lambda_returns
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
        gammas = gammas * torch.ones(rewards.size(), dtype=rewards.dtype)
    if not isinstance(lambda_, torch.Tensor):
        lambda_ = lambda_ * torch.ones(rewards.size(), dtype=rewards.dtype)
    bootstrap_values_tp1 = bootstrap_values[1:, :]
    return multistep_forward_view(rewards, gammas, bootstrap_values_tp1, lambda_, need_grad=need_grad)


def td_lambda_loss(rewards, values, gamma=1.0, lambda_=0.8):
    r"""
    Overview:
        Computing TD($\lambda$) loss given constant gamma and lambda.
        There is no special handling for terminal state value,
        if some state has reached the terminal, just fill in zeros for values and rewards beyond terminal
        (*including the terminal state*, values[terminal] should also be 0)
    Arguments:
        - rewards (:obj:`torch.Tensor`): the returns from time step 0 to T-1, of size [T_traj, batchsize]
        - values (:obj:`torch.Tensor`): estimation of the state value at step 0 to T, of size [T_traj+1, batchsize]
        - gamma (:obj:`float`): constant gamma
        - lambda_ (:obj:`float`): constant lambda (between 0 to 1)
    Returns:
        - loss (:obj:`torch.Tensor`): Computed MSE loss, averaged over the batch, of size []
    """
    returns = generalized_lambda_returns(rewards, gamma, values, lambda_, False)
    # discard the value at T as it should be considered in the next slice
    loss = 0.5 * torch.pow(returns - values[:-1], 2).mean()
    return loss


def compute_importance_weights(
    action_logits, current_logits, action, clipping=lambda x: torch.clamp(x, min=1), eps=1e-8, need_grad=False
):
    r"""
    Overview:
        Computing UPGO loss given constant gamma and lambda. There is no special handling for terminal state value.
        If zeros are passed in, output is not defined but will not be nans.
    Arguments:
        - action_logits (:obj:`torch.Tensor`):
          the logits used producing the trajectory, of size [T_traj, batchsize, n_action_type]
        - current_logits (:obj:`torch.Tensor`): the logits computed by the target policy network,
          of size [T_traj, batchsize, n_action_type]
        - action (:obj:`torch.Tensor`): the chosen action(index) in trajectory, of size [T_traj, batchsize]
        - clipping: the clipping funtion for the importance weight,
          the raw IW tensor of size [T_traj, batchsize] is passed to the function, should return clipped value
        - eps (:obj:`float`): for numerical stability
    Returns:
        - rhos (:obj:`torch.Tensor`): Importance weight, of size [T_traj, batchsize]
    """
    grad_mode = torch.is_grad_enabled()
    torch.set_grad_enabled(need_grad)

    rhos = F.softmax(current_logits, dim=2).gather(2, action.long().unsqueeze(2)).squeeze(2)\
        / (F.softmax(action_logits, dim=2).gather(2, action.long().unsqueeze(2)).squeeze(2)+eps)
    rhos = clipping(rhos)

    torch.set_grad_enabled(grad_mode)
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
    return generalized_lambda_returns(rewards, 1.0, bootstrap_values, lambdas, False)


def upgo_loss(current_logits, rhos, action, rewards, bootstrap_values):
    r"""
    Overview:
        Computing UPGO loss given constant gamma and lambda. There is no special handling for terminal state value,
        if the last state in trajectory is the terminal, just pass a 0 as bootstrap_terminal_value.
    Arguments:
        - current_logits (:obj:`torch.Tensor`): the logits computed by the target policy network,
          of size [T_traj, batchsize, n_action_type]
        - rhos (:obj:`torch.Tensor`): the importance sampling ratio, of size [T_traj, batchsize]
        - action (:obj:`torch.Tensor`): the action taken, of size [T_traj, batchsize]
        - rewards (:obj:`torch.Tensor`): the returns from time step 0 to T-1, of size [T_traj, batchsize]
        - bootstrap_values (:obj:`torch.Tensor`): estimation of the state value at step 0 to T,
          of size [T_traj+1, batchsize]
    Returns:
        - loss (:obj:`torch.Tensor`): Computed importance sampled UPGO loss, averaged over the samples, of size []
    """
    returns = upgo_returns(rewards, bootstrap_values)
    # discard the value at T as it should be considered in the next slice
    advantages = rhos * (returns - bootstrap_values[:-1])
    advantages = advantages.detach()  # to make sure
    losses = advantages * \
        F.log_softmax(current_logits, dim=2).gather(
            2, action.long().unsqueeze(2)).squeeze(2)
    return -losses.mean()


def vtrace_advantages(clipped_rhos, clipped_cs, rewards, bootstrap_values, gammas=1.0, lambda_=0.8, need_grad=False):
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
    grad_mode = torch.is_grad_enabled()
    torch.set_grad_enabled(need_grad)

    if not isinstance(gammas, torch.Tensor):
        gammas = gammas * torch.ones(rewards.size(), dtype=rewards.dtype)
    if not isinstance(lambda_, torch.Tensor):
        lambda_ = lambda_ * torch.ones(rewards.size(), dtype=rewards.dtype)
    deltas = clipped_rhos * \
        (rewards + gammas *
         bootstrap_values[1:, :] - bootstrap_values[:-1, :])  # from 0 to T-1
    result = torch.empty(rewards.size())
    result[-1, :] = bootstrap_values[-2, :] + deltas[-1, :]
    for t in reversed(range(rewards.size()[0] - 1)):
        result[t, :] = bootstrap_values[t, :] + deltas[t, :]\
            + gammas[t, :] * lambda_[t, :] * clipped_cs[t, :] * \
            (result[t+1, :] - bootstrap_values[t+1, :])

    torch.set_grad_enabled(grad_mode)
    return result


def vtrace_loss(current_logits, rhos, cs, action, rewards, bootstrap_values, gamma=1.0, lambda_=0.8):
    r"""
    Overview:
        Computing UPGO loss given constant gamma and lambda. There is no special handling for terminal state value,
        if the last state in trajectory is the terminal, just pass a 0 as bootstrap_terminal_value.
    Arguments:
        - current_logits (:obj:`torch.Tensor`): the logits computed by the target policy network,
          of size [T_traj, batchsize, n_action_type]
        - rhos (:obj:`torch.Tensor`): the clipped importance sampling ratio $\rho$, of size [T_traj, batchsize]
        - cs (:obj:`torch.Tensor`): the clipped importance sampling ratio c, of size [T_traj, batchsize]
        - action (:obj:`torch.Tensor`): the action taken, of size [T_traj, batchsize]
        - rewards (:obj:`torch.Tensor`): the returns from time step 0 to T-1, of size [T_traj, batchsize]
        - bootstrap_values (:obj:`torch.Tensor`): estimation of the state value at step 0 to T,
          of size [T_traj+1, batchsize]
    Returns:
        - loss (:obj:`torch.Tensor`): Computed V-trace loss, averaged over the samples, of size []
    """
    advantages = vtrace_advantages(rhos, cs, rewards, bootstrap_values, gammas=gamma, lambda_=lambda_)
    advantages = advantages.detach()  # to make sure
    losses = advantages * \
        F.log_softmax(current_logits, dim=2).gather(
            2, action.long().unsqueeze(2)).squeeze(2)
    return -losses.mean()


def entropy(policy_logits):
    r"""
    Overview:
        Computing entropy given the logits
    Arguments:
        - policy_logits (:obj:`torch.Tensor`): the logits computed by policy network,
          of size [T_traj, batchsize, n_action_type]
    Returns:
        - entropy (:obj:`torch.Tensor`): Computed entropy, averaged over the samples, of size []
    """
    valid_flag = torch.max(torch.abs(policy_logits), 2)[0] > 0  # exclude all zero logits
    ent_step = -torch.sum(F.softmax(policy_logits, dim=2) * F.log_softmax(policy_logits, dim=2), dim=2)
    numel = policy_logits.size()[0] * policy_logits.size()[1]
    ent = torch.sum(ent_step * valid_flag) * numel / torch.sum(valid_flag)
    # Normalize by actions available.
    normalized_entropy = ent / \
        torch.log(torch.tensor(policy_logits.size()[2]).float())
    return normalized_entropy


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
        - clipping: the clipping funtion for the importance weight,
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


if __name__ == '__main__':
    test_data = {}
    test_data['T_traj'] = 64
    test_data['batchsize'] = 32
    test_data['n_action_type'] = 3
    torch.manual_seed(10)
    test_data['action_logits'] = torch.cat(
        (
            torch.ones((test_data['T_traj'] - 2, test_data['batchsize'], test_data['n_action_type'])),
            torch.zeros((2, test_data['batchsize'], test_data['n_action_type']))
        )
    )
    test_data['current_logits'] = torch.cat(
        (
            torch.ones((test_data['T_traj'] - 2, test_data['batchsize'], test_data['n_action_type'])),
            torch.zeros((2, test_data['batchsize'], test_data['n_action_type']))
        )
    )
    test_data['action'] = torch.randint(0, test_data['n_action_type'], (test_data['T_traj'], test_data['batchsize']))
    test_data['rewards'] = torch.cat(
        (torch.ones((test_data['T_traj'] - 2, test_data['batchsize'])), torch.zeros((2, test_data['batchsize'])))
    )
    # test_data['rewards'][1,:]=100
    test_data['bootstrap_value'] = torch.cat(
        (torch.ones((test_data['T_traj'] - 2, test_data['batchsize'])), torch.zeros((2 + 1, test_data['batchsize'])))
    )

    test_data['bootstrap_value'][1, :] = -100
    test_data['bootstrap_value'][2, :] = -100
    print(
        pg_loss(
            test_data['action_logits'],
            test_data['current_logits'],
            test_data['action'],
            test_data['rewards'],
            test_data['bootstrap_value'],
            upgo_weight=1,
            ent_weight=0.01
        )
    )
    print(td_lambda_loss(test_data['rewards'], test_data['bootstrap_value'], gamma=1, lambda_=0))
