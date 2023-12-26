#############################################################
# This QGPO model is a modification implementation from https://github.com/ChenDRAG/CEP-energy-guided-diffusion
#############################################################

from typing import List, Dict, Any
import functools
import torch
import numpy as np
from ding.torch_utils import to_device
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy

from ding.model.template.qgpo import marginal_prob_std


@POLICY_REGISTRY.register('qgpo')
class QGPOPolicy(Policy):
    """
       Overview:
            Policy class of QGPO algorithm
            Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning
            https://arxiv.org/abs/2304.12824
    """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='qgpo',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool type) on_policy: Determine whether on-policy or off-policy.
        # on-policy setting influences the behaviour of buffer.
        # Default False in QGPO.
        on_policy=False,
        multi_agent=False,
        model=dict(
            score_net=dict(
                qgpo_critic=dict(
                    # (float) The scale of the energy guidance when training qt.
                    # \pi_{behavior}\exp(f(s,a)) \propto \pi_{behavior}\exp(alpha * Q(s,a))
                    alpha=3,
                    # (float) The scale of the energy guidance when training q0.
                    # \mathcal{T}Q(s,a)=r(s,a)+\mathbb{E}_{s'\sim P(s'|s,a),a'\sim\pi_{support}(a'|s')}Q(s',a')
                    # \pi_{support} \propto \pi_{behavior}\exp(q_alpha * Q(s,a))
                    q_alpha=1,
                ),
            ),
            device='cuda',
            # obs_dim
            # action_dim
        ),
        learn=dict(
            # learning rate for behavior model training
            learning_rate=1e-4,
            # batch size during the training of behavior model
            batch_size=4096,
            # batch size during the training of q value
            batch_size_q=256,
            # number of fake action support
            M=16,
            # number of diffusion time steps
            diffusion_steps=15,
            # training iterations when behavior model is fixed
            behavior_policy_stop_training_iter=600000,
            # training iterations when energy-guided policy begin training
            energy_guided_policy_begin_training_iter=600000,
            # training iterations when q value stop training, default None means no limit
            q_value_stop_training_iter=1100000,
        ),
        eval=dict(
            # energy guidance scale for policy in evaluation
            # \pi_{evaluation} \propto \pi_{behavior}\exp(guidance_scale * alpha * Q(s,a))
            guidance_scale=[0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0],
        ),
    )

    def _init_learn(self) -> None:
        self.cuda = self._cfg.cuda
        if self.cuda:
            self.margin_prob_std_fn = functools.partial(marginal_prob_std, device=self._device)
        self.behavior_model_optimizer = torch.optim.Adam(
            self._model.score_model.parameters(), lr=self._cfg.learn.learning_rate
        )
        self.behavior_policy_stop_training_iter = self._cfg.learn.behavior_policy_stop_training_iter if hasattr(
            self._cfg.learn, 'behavior_policy_stop_training_iter'
        ) else np.inf
        self.energy_guided_policy_begin_training_iter = self._cfg.learn.energy_guided_policy_begin_training_iter if hasattr(
            self._cfg.learn, 'energy_guided_policy_begin_training_iter'
        ) else 0
        self.q_value_stop_training_iter = self._cfg.learn.q_value_stop_training_iter if hasattr(
            self._cfg.learn, 'q_value_stop_training_iter'
        ) and self._cfg.learn.q_value_stop_training_iter >= 0 else np.inf

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        if self.cuda:
            data = {k: d.to(self._device) for k, d in data.items()}
        else:
            data = {k: d for k, d in data.items()}
        s = data['s']
        a = data['a']

        # training behavior model
        if self.behavior_policy_stop_training_iter > 0:
            self._model.score_model.condition = s
            behavior_model_training_loss = self._model.loss_fn(a, self.margin_prob_std_fn)
            self.behavior_model_optimizer.zero_grad()
            behavior_model_training_loss.backward()
            self.behavior_model_optimizer.step()
            self._model.score_model.condition = None
            self.behavior_policy_stop_training_iter -= 1
            behavior_model_training_loss = behavior_model_training_loss.detach().cpu().numpy()
        else:
            behavior_model_training_loss = 0

        # training Q function
        self.energy_guided_policy_begin_training_iter -= 1
        self.q_value_stop_training_iter -= 1
        if self.energy_guided_policy_begin_training_iter < 0:
            if self.q_value_stop_training_iter > 0:
                q0_loss = self._model.score_model.q[0].update_q0(data)
            else:
                q0_loss = 0
            qt_loss = self._model.score_model.q[0].update_qt(data)
        else:
            q0_loss = 0
            qt_loss = 0

        total_loss = behavior_model_training_loss + q0_loss + qt_loss

        return dict(
            total_loss=total_loss,
            behavior_model_training_loss=behavior_model_training_loss,
            q0_loss=q0_loss,
            qt_loss=qt_loss,
        )

    def _init_collect(self) -> None:
        pass

    def _forward_collect(self) -> None:
        pass

    def _init_eval(self) -> None:
        self.guidance_scale = self._cfg.eval.guidance_scale
        self.diffusion_steps = self._cfg.eval.diffusion_steps

    def _forward_eval(self, data: dict) -> dict:
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        states = data
        actions = self._model.score_model.select_actions(states, diffusion_steps=self.diffusion_steps)
        output = actions

        return {i: {"action": d} for i, d in zip(data_id, output)}

    def _get_train_sample(self) -> None:
        pass

    def _process_transition(self) -> None:
        pass

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._model.state_dict(),
            'behavior_model_optimizer': self.behavior_model_optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._model.load_state_dict(state_dict['model'])
        self.behavior_model_optimizer.load_state_dict(state_dict['behavior_model_optimizer'])

    def _monitor_vars_learn(self) -> List[str]:
        return ['total_loss', 'behavior_model_training_loss', 'q0_loss', 'qt_loss']
