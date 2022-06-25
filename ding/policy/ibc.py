from typing import Dict, Any, List, Tuple
from collections import namedtuple

from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch import nn

from easydict import EasyDict

from ding.torch_utils import to_device
from ding.utils.data import default_collate, default_decollate
from ding.utils import POLICY_REGISTRY, OPTIMIZER_REGISTRY
from ding.policy.common_utils import default_preprocess_learn
from ding.policy import BehaviourCloningPolicy


# TODO: change to BehavioralCloning or BehaviorCloning
@POLICY_REGISTRY.register('ibc')
class ImplicitBehaviourCloningPolicy(BehaviourCloningPolicy):

    config = dict(
        type='ibc',
        cuda=False,
        on_policy=False,
        continuous=True,
        model=dict(
            type='ebm', 
            type='arebm',
            optimizer=dict(
                type='mcmc',
                type='dfo',
                noise_scale: float
                noise_shrink: float
                iters: int
                train_samples: int
                inference_samples: int
                bounds: np.ndarray
            )
        ),
        learn=dict(
            # multi_gpu=False,
            # update_per_collect=1,
            batch_size=32,
            optim=dict(
                learning_rate=1e-5,
                weight_decay=0.0,
                beta1=0.9,
                beta2=0.999,
                lr_scheduler_step=100,
                lr_scheduler_gamma=0.99,
            )
        ),
        collect=dict(unroll_len=1, ),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=10000, )),
    )
    
    def _init_learn(self):
        optim_cfg = self._cfg.optim  
        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=optim_cfg.learning_rate,
            weight_decay=optim_cfg.weight_decay,
            betas=(optim_cfg.beta1, optim_cfg.beta2),
        )
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer,
            step_size=optim_cfg.lr_scheduler_step,
            gamma=optim_cfg.lr_scheduler_gamma,
        )

    def default_model(self) -> Tuple[str, List[str]]:
        pass
    
    def _forward_learn(self, data):
        if not isinstance(data, dict):
            data = default_collate(data)
        if self._cuda:
            data = to_device(data, self._device)
        self._learn_model.train()

    def _monitor_vars_learn(self):
        pass

    def _init_eval(self):
        pass

    @torch.no_grad()
    def _forward_eval(self, data):
        pass

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
