from typing import List, Dict, Any, Tuple, Union
import torch
import treetensor.torch as ttorch

from ding.torch_utils import SGD
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from .base_policy import Policy


class ModifiedCrossEntropyLoss(torch.nn.Module):

    def __init__(self, reduction: str = 'none'):
        assert reduction == 'none', reduction
        self.reduction = reduction

    def forward(self, inputs, target):
        return -(torch.log_softmax(inputs, dim=-1) * target).sum(dim=-1)


@POLICY_REGISTRY.register('muzero')
class MuZero(Policy):
    """
    Overview:
        MuZero
        EfficientZero
    """
    config = dict(
        type='muzero',
        cuda=False,
        on_policy=False,
        priority=True,
        priority_IS_weight=True,
        batch_size=256,
        discount_factor=0.997,
        learning_rate=1e-3,
        momentum=0.9,
        weight_decay=1e-4,
        grad_clip_type='clip_norm',
        grad_clip_value=5,
        policy_weight=1.0,
        value_weight=0.25,
        consistent_weight=1.0,
        value_prefix_weight=2.0,
        image_unroll_len=5,
        lstm_horizon_len=5,
    )

    def _init_learn(self) -> None:
        self._metric_loss = torch.nn.L1Loss()
        self._cos = torch.nn.CosineSimilarity(dim=1, eps=1e-05)
        self._ce = ModifiedCrossEntropyLoss(reduction='none')
        self._optimizer = SGD(
            self._model.parameters(),
            lr=self._cfg.learning_rate,
            momentum=self._cfg.momentum,
            weight_decay=self._cfg.weight_decay,
            grad_clip_type=self._cfg.grad_clip_type,
            clip_value=self._cfg.grad_clip_value
        )

        self._learn_model = model_wrap(self._model, wrapper='base')
        self._learn_model.reset()

    def _data_preprocess_learn(self, data: ttorch.Tensor):
        # TODO data augmentation before learning
        data = data.cuda(self.device)
        data = ttorch.stack(data)
        return data

    def _forward_learn(self, data: ttorch.Tensor) -> Dict[str, Union[float, int]]:
        self._learn_model.train()
        losses = ttorch.as_tensor({})
        losses.consistent_loss = torch.zeros(1).to(self.device)
        losses.value_prefix_loss = torch.zeros(1).to(self.device)

        # first step
        output = self._learn_model.forward(data.obs, mode='init')
        losses.value_loss = self._ce(output.value, data.target_value[0])
        td_error_per_sample = losses.value_loss.clone().detach()
        losses.policy_loss = self._ce(output.logit, data.target_action[0])

        # unroll N step
        N = self._cfg.image_unroll_len
        for i in range(N):
            if self._cfg.value_prefix_weight > 0:
                output = self._learn_model.forward(
                    output.hidden_state, output.hidden_state_reward, data.action[i], mode='recurrent'
                )
            else:
                output = self._learn_model.forward(output.hidden_state, data.action[i], mode='recurrent')
            losses.value_loss += self._ce(output.value, data.target_value[i + 1])
            losses.policy_loss += self._ce(output.logit, data.target_action[i + 1])
            # consistent loss
            if self._cfg.consistent_weight > 0:
                with torch.no_grad():
                    next_hidden_state = self._learn_model.forward(data.next_obs, mode='init')
                    projected_next = self._learn_model.forward(next_hidden_state, mode='project')
                projected_now = self._learn_model.forward(output.hidden_state, mode='project')
                losses.consistent_loss += -(self._cos(projected_now, projected_next) * data.mask[i])
            # value prefix loss
            if self._cfg.value_prefix_weight > 0:
                losses.value_prefix_loss += self._ce(output.value_prefix, data.target_value_prefix[i])
            # set half gradient
            output.hidden_state.register_hook(lambda grad: grad * 0.5)
            # reset hidden states
            if (i + 1) % self._cfg.lstm_horizon_len == 0:
                output.hidden_state_reward.zero_()

        total_loss = (
            self._cfg.policy_weight * losses.policy_loss + self._cfg.value_weight * losses.value_loss +
            self._cfg.value_prefix_weight * losses.value_prefix_loss +
            self._cfg.consistent_weight * losses.consistent_loss
        )
        total_loss = total_loss.mean()
        total_loss.register_hook(lambda grad: grad / N)

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'priority': td_error_per_sample.abs().tolist(),
        }.update({k: v.mean().item()
                  for k, v in losses.items()})
