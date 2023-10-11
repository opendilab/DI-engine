from typing import List, Dict, Any, Tuple, Union, Callable, Optional
from collections import namedtuple
from easydict import EasyDict
import copy
import random
import numpy as np
import torch
import treetensor.torch as ttorch
from torch.optim import AdamW

from ding.rl_utils import ppo_data, ppo_error, ppo_policy_error, ppo_policy_data, gae, gae_data, ppo_error_continuous, \
    get_gae, ppo_policy_error_continuous, ArgmaxSampler, MultinomialSampler, ReparameterizationSampler, MuSampler, \
    HybridStochasticSampler, HybridDeterminsticSampler, value_transform, value_inv_transform, symlog, inv_symlog
from ding.utils import POLICY_REGISTRY, RunningMeanStd


@POLICY_REGISTRY.register('ppof')
class PPOFPolicy:
    config = dict(
        type='ppo',
        on_policy=True,
        cuda=True,
        action_space='discrete',
        discount_factor=0.99,
        gae_lambda=0.95,
        # learn
        epoch_per_collect=10,
        batch_size=64,
        learning_rate=3e-4,
        # learningrate scheduler, which the format is (10000, 0.1)
        lr_scheduler=None,
        weight_decay=0,
        value_weight=0.5,
        entropy_weight=0.01,
        clip_ratio=0.2,
        adv_norm=True,
        value_norm='baseline',
        ppo_param_init=True,
        grad_norm=0.5,
        # collect
        n_sample=128,
        unroll_len=1,
        # eval
        deterministic_eval=True,
        # model
        model=dict(),
    )
    mode = ['learn', 'collect', 'eval']

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    @classmethod
    def default_model(cls: type) -> Callable:
        from .model import PPOFModel
        return PPOFModel

    def __init__(self, cfg: "EasyDict", model: torch.nn.Module, enable_mode: List[str] = None) -> None:
        self._cfg = cfg
        if model is None:
            self._model = self.default_model()
        else:
            self._model = model
        if self._cfg.cuda and torch.cuda.is_available():
            self._device = 'cuda'
            self._model.cuda()
        else:
            self._device = 'cpu'
        assert self._cfg.action_space in ["continuous", "discrete", "hybrid", 'multi_discrete']
        self._action_space = self._cfg.action_space
        if self._cfg.ppo_param_init:
            self._model_param_init()

        if enable_mode is None:
            enable_mode = self.mode
        self.enable_mode = enable_mode
        if 'learn' in enable_mode:
            self._optimizer = AdamW(
                self._model.parameters(),
                lr=self._cfg.learning_rate,
                weight_decay=self._cfg.weight_decay,
            )
            # define linear lr scheduler
            if self._cfg.lr_scheduler is not None:
                epoch_num, min_lr_lambda = self._cfg.lr_scheduler

                self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self._optimizer,
                    lr_lambda=lambda epoch: max(1.0 - epoch * (1.0 - min_lr_lambda) / epoch_num, min_lr_lambda)
                )

            if self._cfg.value_norm:
                self._running_mean_std = RunningMeanStd(epsilon=1e-4, device=self._device)
        if 'collect' in enable_mode:
            if self._action_space == 'discrete':
                self._collect_sampler = MultinomialSampler()
            elif self._action_space == 'continuous':
                self._collect_sampler = ReparameterizationSampler()
            elif self._action_space == 'hybrid':
                self._collect_sampler = HybridStochasticSampler()
        if 'eval' in enable_mode:
            if self._action_space == 'discrete':
                if self._cfg.deterministic_eval:
                    self._eval_sampler = ArgmaxSampler()
                else:
                    self._eval_sampler = MultinomialSampler()
            elif self._action_space == 'continuous':
                if self._cfg.deterministic_eval:
                    self._eval_sampler = MuSampler()
                else:
                    self._eval_sampler = ReparameterizationSampler()
            elif self._action_space == 'hybrid':
                if self._cfg.deterministic_eval:
                    self._eval_sampler = HybridDeterminsticSampler()
                else:
                    self._eval_sampler = HybridStochasticSampler()
        # for compatibility
        self.learn_mode = self
        self.collect_mode = self
        self.eval_mode = self

    def _model_param_init(self):
        for n, m in self._model.named_modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        if self._action_space in ['continuous', 'hybrid']:
            for m in list(self._model.critic.modules()) + list(self._model.actor.modules()):
                if isinstance(m, torch.nn.Linear):
                    # orthogonal initialization
                    torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    torch.nn.init.zeros_(m.bias)
            # init log sigma
            if self._action_space == 'continuous':
                torch.nn.init.constant_(self._model.actor_head.log_sigma_param, -0.5)
                for m in self._model.actor_head.mu.modules():
                    if isinstance(m, torch.nn.Linear):
                        torch.nn.init.zeros_(m.bias)
                        m.weight.data.copy_(0.01 * m.weight.data)
            elif self._action_space == 'hybrid':  # actor_head[1]: ReparameterizationHead, for action_args
                if hasattr(self._model.actor_head[1], 'log_sigma_param'):
                    torch.nn.init.constant_(self._model.actor_head[1].log_sigma_param, -0.5)
                    for m in self._model.actor_head[1].mu.modules():
                        if isinstance(m, torch.nn.Linear):
                            torch.nn.init.zeros_(m.bias)
                            m.weight.data.copy_(0.01 * m.weight.data)

    def forward(self, data: ttorch.Tensor) -> Dict[str, Any]:
        return_infos = []
        self._model.train()
        bs = self._cfg.batch_size
        data = data[:self._cfg.n_sample // bs * bs]  # rounding

        # outer training loop
        for epoch in range(self._cfg.epoch_per_collect):
            # recompute adv
            with torch.no_grad():
                # get the value dictionary
                # In popart, the dictionary has two keys: 'pred' and 'unnormalized_pred'
                value = self._model.compute_critic(data.obs)
                next_value = self._model.compute_critic(data.next_obs)
                reward = data.reward

                assert self._cfg.value_norm in ['popart', 'value_rescale', 'symlog', 'baseline'],\
                    'Not supported value normalization! Value normalization supported: \
                        popart, value rescale, symlog, baseline'

                if self._cfg.value_norm == 'popart':
                    unnormalized_value = value['unnormalized_pred']
                    unnormalized_next_value = value['unnormalized_pred']

                    mu = self._model.critic_head.popart.mu
                    sigma = self._model.critic_head.popart.sigma
                    reward = (reward - mu) / sigma

                    value = value['pred']
                    next_value = next_value['pred']
                elif self._cfg.value_norm == 'value_rescale':
                    value = value_inv_transform(value['pred'])
                    next_value = value_inv_transform(next_value['pred'])
                elif self._cfg.value_norm == 'symlog':
                    value = inv_symlog(value['pred'])
                    next_value = inv_symlog(next_value['pred'])
                elif self._cfg.value_norm == 'baseline':
                    value = value['pred'] * self._running_mean_std.std
                    next_value = next_value['pred'] * self._running_mean_std.std

                traj_flag = data.get('traj_flag', None)  # traj_flag indicates termination of trajectory
                adv_data = gae_data(value, next_value, reward, data.done, traj_flag)
                data.adv = gae(adv_data, self._cfg.discount_factor, self._cfg.gae_lambda)

                unnormalized_returns = value + data.adv  # In popart, this return is normalized

                if self._cfg.value_norm == 'popart':
                    self._model.critic_head.popart.update_parameters((data.reward).unsqueeze(1))
                elif self._cfg.value_norm == 'value_rescale':
                    value = value_transform(value)
                    unnormalized_returns = value_transform(unnormalized_returns)
                elif self._cfg.value_norm == 'symlog':
                    value = symlog(value)
                    unnormalized_returns = symlog(unnormalized_returns)
                elif self._cfg.value_norm == 'baseline':
                    value /= self._running_mean_std.std
                    unnormalized_returns /= self._running_mean_std.std
                    self._running_mean_std.update(unnormalized_returns.cpu().numpy())
                data.value = value
                data.return_ = unnormalized_returns

            # inner training loop
            split_data = ttorch.split(data, self._cfg.batch_size)
            random.shuffle(list(split_data))
            for batch in split_data:
                output = self._model.compute_actor_critic(batch.obs)
                adv = batch.adv
                if self._cfg.adv_norm:
                    # Normalize advantage in a train_batch
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Calculate ppo error
                if self._action_space == 'continuous':
                    ppo_batch = ppo_data(
                        output.logit, batch.logit, batch.action, output.value, batch.value, adv, batch.return_, None
                    )
                    ppo_loss, ppo_info = ppo_error_continuous(ppo_batch, self._cfg.clip_ratio)
                elif self._action_space == 'discrete':
                    ppo_batch = ppo_data(
                        output.logit, batch.logit, batch.action, output.value, batch.value, adv, batch.return_, None
                    )
                    ppo_loss, ppo_info = ppo_error(ppo_batch, self._cfg.clip_ratio)
                elif self._action_space == 'hybrid':
                    # discrete part (discrete policy loss and entropy loss)
                    ppo_discrete_batch = ppo_policy_data(
                        output.logit.action_type, batch.logit.action_type, batch.action.action_type, adv, None
                    )
                    ppo_discrete_loss, ppo_discrete_info = ppo_policy_error(ppo_discrete_batch, self._cfg.clip_ratio)
                    # continuous part (continuous policy loss and entropy loss, value loss)
                    ppo_continuous_batch = ppo_data(
                        output.logit.action_args, batch.logit.action_args, batch.action.action_args, output.value,
                        batch.value, adv, batch.return_, None
                    )
                    ppo_continuous_loss, ppo_continuous_info = ppo_error_continuous(
                        ppo_continuous_batch, self._cfg.clip_ratio
                    )
                    # sum discrete and continuous loss
                    ppo_loss = type(ppo_continuous_loss)(
                        ppo_continuous_loss.policy_loss + ppo_discrete_loss.policy_loss, ppo_continuous_loss.value_loss,
                        ppo_continuous_loss.entropy_loss + ppo_discrete_loss.entropy_loss
                    )
                    ppo_info = type(ppo_continuous_info)(
                        max(ppo_continuous_info.approx_kl, ppo_discrete_info.approx_kl),
                        max(ppo_continuous_info.clipfrac, ppo_discrete_info.clipfrac)
                    )
                wv, we = self._cfg.value_weight, self._cfg.entropy_weight
                total_loss = ppo_loss.policy_loss + wv * ppo_loss.value_loss - we * ppo_loss.entropy_loss

                self._optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._cfg.grad_norm)
                self._optimizer.step()

                return_info = {
                    'cur_lr': self._optimizer.defaults['lr'],
                    'total_loss': total_loss.item(),
                    'policy_loss': ppo_loss.policy_loss.item(),
                    'value_loss': ppo_loss.value_loss.item(),
                    'entropy_loss': ppo_loss.entropy_loss.item(),
                    'adv_max': adv.max().item(),
                    'adv_mean': adv.mean().item(),
                    'value_mean': output.value.mean().item(),
                    'value_max': output.value.max().item(),
                    'approx_kl': ppo_info.approx_kl,
                    'clipfrac': ppo_info.clipfrac,
                }
                if self._action_space == 'continuous':
                    return_info.update(
                        {
                            'action': batch.action.float().mean().item(),
                            'mu_mean': output.logit.mu.mean().item(),
                            'sigma_mean': output.logit.sigma.mean().item(),
                        }
                    )
                elif self._action_space == 'hybrid':
                    return_info.update(
                        {
                            'action': batch.action.action_args.float().mean().item(),
                            'mu_mean': output.logit.action_args.mu.mean().item(),
                            'sigma_mean': output.logit.action_args.sigma.mean().item(),
                        }
                    )
                return_infos.append(return_info)

        if self._cfg.lr_scheduler is not None:
            self._lr_scheduler.step()

        return return_infos

    def state_dict(self) -> Dict[str, Any]:
        state_dict = {
            'model': self._model.state_dict(),
        }
        if 'learn' in self.enable_mode:
            state_dict['optimizer'] = self._optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._model.load_state_dict(state_dict['model'])
        if 'learn' in self.enable_mode:
            self._optimizer.load_state_dict(state_dict['optimizer'])

    def collect(self, data: ttorch.Tensor) -> ttorch.Tensor:
        self._model.eval()
        with torch.no_grad():
            output = self._model.compute_actor_critic(data)
            action = self._collect_sampler(output.logit)
            output.action = action
        return output

    def process_transition(self, obs: ttorch.Tensor, inference_output: dict, timestep: namedtuple) -> ttorch.Tensor:
        return ttorch.as_tensor(
            {
                'obs': obs,
                'next_obs': timestep.obs,
                'action': inference_output.action,
                'logit': inference_output.logit,
                'value': inference_output.value,
                'reward': timestep.reward,
                'done': timestep.done,
            }
        )

    def eval(self, data: ttorch.Tensor) -> ttorch.Tensor:
        self._model.eval()
        with torch.no_grad():
            logit = self._model.compute_actor(data)
            action = self._eval_sampler(logit)
        return ttorch.as_tensor({'logit': logit, 'action': action})

    def monitor_vars(self) -> List[str]:
        variables = [
            'cur_lr',
            'policy_loss',
            'value_loss',
            'entropy_loss',
            'adv_max',
            'adv_mean',
            'approx_kl',
            'clipfrac',
            'value_max',
            'value_mean',
        ]
        if self._action_space in ['action', 'mu_mean', 'sigma_mean']:
            variables += ['mu_mean', 'sigma_mean', 'action']
        return variables

    def reset(self, env_id_list: Optional[List[int]] = None) -> None:
        pass
