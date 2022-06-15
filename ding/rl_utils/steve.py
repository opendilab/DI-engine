from typing import List, Callable
import torch


def steve_target(
        obs, reward, done, q_fn, rew_fn, policy_fn, transition_fn, done_fn, rollout_step, discount_factor, ensemble_num
) -> torch.Tensor:
    """
    Shapes:
        - obs (:obj:`torch.Tensor`): :math:`(B, N1)`, where N1 is obs shape
        - reward (:obj:`torch.Tensor`): :math:`(B, )`
        - done (:obj:`torch.Tensor`): :math:`(B, )`
        - return_ (:obj:`torch.Tensor`): :math:`(B, )`
    """
    # tile first data
    ensemble_q_num, ensemble_r_num, ensemble_d_num = ensemble_num
    obs = obs.unsqueeze(1).repeat(1, ensemble_q_num, 1)
    reward = reward.unsqueeze(1).repeat(1, ensemble_r_num)
    done = done.unsqueeze(1).repeat(1, ensemble_d_num)

    with torch.no_grad():
        device = reward.device
        # real data
        action = policy_fn(obs)
        q_value = q_fn(obs, action)
        q_list, r_list, d_list = [q_value], [reward], [done]
        # imagination data
        for i in range(rollout_step):
            next_obs = transition_fn(obs, action)
            reward = rew_fn(obs, action, next_obs)
            done = done_fn(obs, action, next_obs)
            obs = next_obs
            action = policy_fn(obs)
            q_value = q_fn(obs, action)

            q_list.append(q_value)
            r_list.append(reward)
            d_list.append(done)

    q_value = torch.stack(q_list, dim=1)  # B, H, M
    reward = torch.stack(r_list, dim=1)  # B, H, N
    done = torch.stack(d_list, dim=1)  # B, H, L

    H = rollout_step + 1

    time_factor = torch.full((1, H), discount_factor).to(device) ** torch.arange(float(H))
    upper_tri = torch.triu(torch.randn(H, H)).to(device)
    reward_coeff = (upper_tri * time_factor).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # 1, H, H, 1, 1
    value_coeff = (time_factor ** discount_factor).unsqueeze(-1).unsqueeze(-1)  # 1, H, 1, 1

    # (B, 1, L) cat (B, H-1, L)
    reward_done = torch.cat([torch.ones_like(done[:, 0]).unsqueeze(1), torch.cumprod(done[:, :-1], dim=1)], dim=1)
    reward_done = reward_done.unsqueeze(1).unsqueeze(-2)
    value_done = done.unsqueeze(-2)

    # (B, 1, H, N, 1) x (1, H, H, 1, 1) x (B, 1, H, 1, L)
    reward = reward.unsqueeze(1).unsqueeze(-1) * reward_coeff * reward_done
    # (B, H, N, L)
    cum_reward = torch.sum(reward, dim=2)  # sum in the second H
    # (B, H, M, 1) x (1, H, 1, 1) x (B, H, 1, L) = B, H, M, L
    q_value = q_value.unsqueeze(-1) * value_coeff * value_done
    # (B, H, 1, N, L) + (B, H, M, 1, L) = B, H, M, N, L
    target = cum_reward.unsqueeze(-3) + q_value.unsqueeze(-2)

    target = target.view(target.shape[0], H, -1)  # B, H, MxNxL
    target_mean = target.mean(-1)
    target_var = target.var(-1)
    target_confidence = 1. / (target_var + 1e-8)  # B, H
    target_confidence = torch.clamp(target_confidence, 0, 100)  # for some special case
    # norm
    target_confidence /= target_confidence.sum(dim=1, keepdim=True)

    return_ = (target_mean * target_confidence).sum(dim=1)
    return return_
