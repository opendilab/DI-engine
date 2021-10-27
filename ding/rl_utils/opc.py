import torch

def compute_q_opc(
        q_values: torch.Tensor,
        v_pred: torch.Tensor,
        rewards: torch.Tensor,
        actions: torch.Tensor,
        weights: torch.Tensor,
        gamma: float = 0.9
) -> torch.Tensor:
    rewards = rewards.unsqueeze(-1)  # shape T,B,1
    actions = actions.unsqueeze(-1)  # shape T,B,1
    weights = weights.unsqueeze(-1)  # shape T,B,1
    q_opc = torch.zeros_like(v_pred)  # shape (T+1),B,1
    n_len = q_opc.size()[0]  # T+1
    tmp_opc = v_pred[-1, ...]  # shape B,1
    q_opc[-1, ...] = v_pred[-1, ...]
    q_values = q_values[0:-1, ...]  # shape T,B,1

    for idx in reversed(range(n_len - 1)):
        q_opc[idx, ...] = rewards[idx, ...] + gamma * weights[idx, ...] * tmp_opc
        tmp_opc = (q_opc[idx, ...] - q_values[idx, ...]) + v_pred[idx, ...]
    return q_opc  # shape (T+1),B,1


