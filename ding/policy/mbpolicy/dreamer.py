from typing import Dict, Any, List
import torch
from torch import nn
from copy import deepcopy
from ding.torch_utils import Adam, to_device
from ding.utils import POLICY_REGISTRY, deep_merge_dicts
from ding.policy import Policy
from ding.rl_utils import generalized_lambda_returns
from ding.model import model_wrap
from ding.policy.common_utils import default_preprocess_learn

from .utils import imagine, compute_target, compute_actor_loss, RewardEMA, tensorstats


@POLICY_REGISTRY.register('dreamer')
class DREAMERPolicy(Policy):
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='dreamer',
        # (bool) Whether to use cuda for network and loss computation.
        cuda=False,
        # (int)
        imag_horizon=15,
        learn=dict(
            # (float) Lambda for TD-lambda return.
            lambda_=0.95,
            # (float) Max norm of gradients.
            grad_clip=100,
            learning_rate=0.001,
            batch_size=256,
            imag_sample=True,
            slow_value_target=True,
            discount=0.997,
            reward_EMA=True,
            actor_entropy=3e-4,
            actor_state_entropy=0.0,
            value_decay=0.0,
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        return 'dreamervac', ['ding.model.template.vac']
    
    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config, main and target models.
        """
        # Algorithm config
        self._lambda = self._cfg.learn.lambda_
        self._grad_clip = self._cfg.learn.grad_clip

        self._critic = self._model.critic
        self._actor = self._model.actor
        
        if self._cfg.learn.slow_value_target:
            self._slow_value = deepcopy(self._critic)
            self._updates = 0
        
        # Optimizer
        self._optimizer_value = Adam(
            self._critic.parameters(),
            lr=self._cfg.learn.learning_rate,
        )
        self._optimizer_actor = Adam(
            self._actor.parameters(),
            lr=self._cfg.learn.learning_rate,
        )
        
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()

        self._forward_learn_cnt = 0

        if self._cfg.learn.reward_EMA:
            self.reward_ema = RewardEMA(device=self._device)

    def _forward_learn(self, start: dict, repeats=None, world_model=None, envstep) -> Dict[str, Any]:
        # log dict
        log_vars = {}
        world_model.requires_grad_(requires_grad=False)
        self._actor.requires_grad_(requires_grad=True)
        # start is dict of {stoch, deter, logit}
        if self._cuda:
            start = to_device(start, self._device)

        self._learn_model.train()
        self._target_model.train()
        
        # train self._actor
        imag_feat, imag_state, imag_action = imagine(
                    self._cfg.learn, world_model, start, self._actor, self._cfg.imag_horizon, repeats
                )
        reward = world_model.heads["reward"](world_model.dynamics.get_feat(imag_state)).mode()
        actor_ent = self._actor(imag_feat).entropy()
        state_ent = world_model.dynamics.get_dist(imag_state).entropy()
        # this target is not scaled
        # slow is flag to indicate whether slow_target is used for lambda-return
        target, weights, base = compute_target(
            self._cfg.learn, world_model, self._critic, imag_feat, imag_state, imag_action, reward, actor_ent, state_ent
        )
        actor_loss, mets = compute_actor_loss(
            self._cfg.learn,
            self._actor,
            self.reward_ema,
            imag_feat,
            imag_state,
            imag_action,
            target,
            actor_ent,
            state_ent,
            weights,
            base,
        )
        log_vars.update(mets)
        value_input = imag_feat
        self._actor.requires_grad_(requires_grad=False)

        self._critic.requires_grad_(requires_grad=True)
        value = self._critic(value_input[:-1].detach())
        # to do
        target = torch.stack(target, dim=1)
        # (time, batch, 1), (time, batch, 1) -> (time, batch)
        value_loss = -value.log_prob(target.detach())
        slow_target = self._slow_value(value_input[:-1].detach())
        if self._cfg.learn.slow_value_target:
            value_loss = value_loss - value.log_prob(
                slow_target.mode().detach()
            )
        if self._cfg.learn.value_decay:
            value_loss += self._cfg.learn.value_decay * value.mode()
        # (time, batch, 1), (time, batch, 1) -> (1,)
        value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])
        self._critic.requires_grad_(requires_grad=False)

        log_vars.update(tensorstats(value.mode(), "value"))
        log_vars.update(tensorstats(target, "target"))
        log_vars.update(tensorstats(reward, "imag_reward"))
        log_vars.update(tensorstats(imag_action, "imag_action"))
        log_vars["actor_ent"] = torch.mean(actor_ent).detach().cpu().numpy()
        # ====================
        # actor-critic update
        # ====================
        self._model.requires_grad_(requires_grad=True)
        
        loss_dict = {
            'critic_loss': value_loss,
            'actor_loss': actor_loss,
        }

        norm_dict = self._update(loss_dict)
        
        self._model.requires_grad_(requires_grad=False)
        # =============
        # after update
        # =============
        self._forward_learn_cnt += 1

        return {
            **log_vars,
            **norm_dict,
            **loss_dict,
        }

    def _update(self, loss_dict):
        # update actor
        self._optimizer_actor.zero_grad()
        loss_dict['actor_loss'].backward()
        actor_norm = nn.utils.clip_grad_norm_(self._model.actor.parameters(), self._grad_clip)
        self._optimizer_actor.step()
        # update critic
        self._optimizer_value.zero_grad()
        loss_dict['critic_loss'].backward()
        critic_norm = nn.utils.clip_grad_norm_(self._model.critic.parameters(), self._grad_clip)
        self._optimizer_value.step()
        return {'actor_grad_norm': actor_norm, 'critic_grad_norm': critic_norm}

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        return [
            'normed_target_mean',
            'normed_target_std',
            'normed_target_min',
            'normed_target_max',
            'EMA_005',
            'EMA_095',
            'actor_entropy',
            'actor_state_entropy',
            'value_mean',
            'value_std',
            'value_min',
            'value_max',
            'target_mean',
            'target_std',
            'target_min',
            'target_max',
            'imag_reward_mean',
            'imag_reward_std',
            'imag_reward_min',
            'imag_reward_max',
            'imag_action_mean',
            'imag_action_std',
            'imag_action_min',
            'imag_action_max',
            'actor_ent',
            'actor_loss',
            'critic_loss',
            'actor_grad_norm',
            'critic_grad_norm'
        ]
