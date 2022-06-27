from typing import Dict, Any, List, Tuple
from collections import namedtuple
from easydict import EasyDict

import torch
import torch.nn.functional as F

from easydict import EasyDict

from ding.model import model_wrap
from ding.torch_utils import to_device
from ding.utils.data import default_collate, default_decollate
from ding.utils import POLICY_REGISTRY
from .bc import BehaviourCloningPolicy
from ding.model.template.ebm import create_stochastic_optimizer, StochasticOptimizer
from ding.torch_utils import unsqueeze_repeat


@POLICY_REGISTRY.register('ibc')
class IBCPolicy(BehaviourCloningPolicy):

    config = dict(
        type='ibc',
        cuda=False,
        on_policy=False,
        continuous=True,
        model=dict(
            # hidden_size=256,
            # hidden_layer_num=2,
            stochastic_optim=dict(
                type='dfo',
                noise_scale=0.33,
                noise_shrink=0.5,
                iters=3,
                train_samples=256,
                inference_samples=512,
                cuda=False,
            )
        ),
        learn=dict(
            train_epoch=30000,
            batch_size=256,
            multi_gpu=False,
            optim=dict(
                learning_rate=1e-5,
                weight_decay=0.0,
                beta1=0.9,
                beta2=0.999,
            ),
        ),
        collect=dict(normalize_states=True,),
        eval=dict(evaluator=dict(eval_freq=10000, )),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        return 'ebm', ['ding.model.template.ebm']
    
    def _init_learn(self):
        optim_cfg = self._cfg.learn.optim
        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=optim_cfg.learning_rate,
            weight_decay=optim_cfg.weight_decay,
            betas=(optim_cfg.beta1, optim_cfg.beta2),
        )
        self._stochastic_optimizer: StochasticOptimizer = \
            create_stochastic_optimizer(self._cfg.model.stochastic_optim)
        self._learn_model = model_wrap(self._model, 'base')
        self._learn_model.reset()
    
    def _forward_learn(self, data):
        if not isinstance(data, dict):
            data = default_collate(data)
        if self._cuda:
            data = to_device(data, self._device)
        self._learn_model.train()

        # obs: (B, O)
        # action: (B, A)
        obs, action = data['obs'], data['action']
        batch_size = obs.shape[0]

        # (B, N, A)
        negatives = self._stochastic_optimizer.sample(batch_size, self._learn_model)

        # (B, N+1, A)
        targets = torch.cat([action.unsqueeze(dim=1), negatives], dim=1)

        permutation = torch.rand(targets.shape[0], targets.shape[1]).argsort(dim=1)
        targets = targets[torch.arange(targets.shape[0]).unsqueeze(-1), permutation]

        ground_truth = (permutation == 0).nonzero()[:, 1].to(self._device)

        # (B, N+1)
        # or (B, N+1, A) for autoregressive ebm
        energy = self._learn_model.forward(obs, targets)

        logits = -1.0 * energy
        if len(logits.shape) == 3:
            # autoregressive case
            # (B, A)
            ground_truth = unsqueeze_repeat(ground_truth, logits.shape[-1], -1)
        loss = F.cross_entropy(logits, ground_truth)

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._optimizer.step()

        return {
            # 'cur_lr': self._scheduler.get_last_lr()[0],
            'total_loss': loss.item(),
        }

    def _monitor_vars_learn(self):
        return [
            # 'cur_lr', 
            'total_loss'
        ]

    def _init_eval(self):
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        self._eval_model.reset()

    @torch.no_grad()
    def _forward_eval(self, data: dict) -> dict:
        tensor_input = isinstance(data, torch.Tensor)
        if tensor_input:
            data = default_collate(list(data))
        else:
            data_id = list(data.keys())
            data = default_collate(list(data.values()))

        if self.cfg.collect.normalize_states:
            data = (data - self._mean) / self._std
        if self._cuda:
            data = to_device(data, self._device)

        self._eval_model.eval()
        # output = self._eval_model.forward(data, mode='compute_actor')
        output = self._stochastic_optimizer.infer(data, self._eval_model)
        output = dict(action=output)

        if self._cuda:
            output = to_device(output, 'cpu')
        if tensor_input:
            return output
        else:
            output = default_decollate(output)
            return {i: d for i, d in zip(data_id, output)}

    def set_norm_statistics(self, statistics: EasyDict) -> None:
        self._mean = statistics.mean
        self._std = statistics.std
        self._stochastic_optimizer.set_action_bounds(statistics.action_bounds)

    # =================================================================== #
    # Implicit Behavioral Cloning does not need `collect`-related functions
    # =================================================================== #
    def _init_collect(self):
        raise NotImplementedError

    def _forward_collect(self, data: Dict[int, Any], eps: float) -> Dict[int, Any]:
        raise NotImplementedError

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        raise NotImplementedError

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError
