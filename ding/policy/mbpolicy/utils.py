from typing import Callable, Tuple, Union
import torch
from torch import Tensor
from ding.torch_utils import fold_batch, unfold_batch
from ding.rl_utils import generalized_lambda_returns
from ding.torch_utils.network.dreamer import static_scan


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


def imagine(cfg, world_model, start, actor, horizon, repeats=None):
    dynamics = world_model.dynamics
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}

    def step(prev, _):
        state, _, _ = prev
        feat = dynamics.get_feat(state)
        inp = feat.detach()
        action = actor(inp).sample()
        succ = dynamics.img_step(state, action, sample=cfg.imag_sample)
        return succ, feat, action

    succ, feats, actions = static_scan(step, [torch.arange(horizon)], (start, None, None))
    states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

    return feats, states, actions


def compute_target(cfg, world_model, critic, imag_feat, imag_state, reward, actor_ent, state_ent):
    if "discount" in world_model.heads:
        inp = world_model.dynamics.get_feat(imag_state)
        discount = cfg.discount * world_model.heads["discount"](inp).mean
        # TODO whether to detach
        discount = discount.detach()
    else:
        discount = cfg.discount * torch.ones_like(reward)

    value = critic(imag_feat).mode()
    # value(imag_horizon, 16*64, 1)
    # action(imag_horizon, 16*64, ch)
    # discount(imag_horizon, 16*64, 1)
    target = generalized_lambda_returns(value, reward[:-1], discount[:-1], cfg.lambda_)
    weights = torch.cumprod(torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0).detach()
    return target, weights, value[:-1]


def compute_actor_loss(
    cfg,
    actor,
    reward_ema,
    imag_feat,
    imag_state,
    imag_action,
    target,
    actor_ent,
    state_ent,
    weights,
    base,
):
    metrics = {}
    inp = imag_feat.detach()
    policy = actor(inp)
    actor_ent = policy.entropy()
    # Q-val for actor is not transformed using symlog
    if cfg.reward_EMA:
        offset, scale = reward_ema(target)
        normed_target = (target - offset) / scale
        normed_base = (base - offset) / scale
        adv = normed_target - normed_base
        metrics.update(tensorstats(normed_target, "normed_target"))
        values = reward_ema.values
        metrics["EMA_005"] = values[0].detach().cpu().numpy().item()
        metrics["EMA_095"] = values[1].detach().cpu().numpy().item()

    actor_target = adv
    if cfg.actor_entropy > 0:
        actor_entropy = cfg.actor_entropy * actor_ent[:-1][:, :, None]
        actor_target += actor_entropy
        metrics["actor_entropy"] = torch.mean(actor_entropy).detach().cpu().numpy().item()
    if cfg.actor_state_entropy > 0:
        state_entropy = cfg.actor_state_entropy * state_ent[:-1]
        actor_target += state_entropy
        metrics["actor_state_entropy"] = torch.mean(state_entropy).detach().cpu().numpy().item()
    actor_loss = -torch.mean(weights[:-1] * actor_target)
    return actor_loss, metrics


class RewardEMA(object):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.values = torch.zeros((2, )).to(device)
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values = self.alpha * x_quantile + (1 - self.alpha) * self.values
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]
        return offset.detach(), scale.detach()


def tensorstats(tensor, prefix=None):
    metrics = {
        'mean': torch.mean(tensor).detach().cpu().numpy(),
        'std': torch.std(tensor).detach().cpu().numpy(),
        'min': torch.min(tensor).detach().cpu().numpy(),
        'max': torch.max(tensor).detach().cpu().numpy(),
    }
    if prefix:
        metrics = {f'{prefix}_{k}': v.item() for k, v in metrics.items()}
    return metrics
