from typing import Dict, Optional, List
from easydict import EasyDict
import os.path as osp
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ding.model import model_wrap
from ding.policy import Policy
from ding.torch_utils import to_device
from ding.rl_utils import td_lambda_data, td_lambda_error, vtrace_data_with_rho, vtrace_error_with_rho, \
    upgo_data, upgo_error
from ding.utils import EasyTimer
from dizoo.distar.model import Model
from .utils import collate_fn_learn

EPS = 1e-9


def entropy_error(target_policy_probs_dict, target_policy_log_probs_dict, mask, head_weights_dict):
    total_entropy_loss = 0.
    entropy_dict = {}
    for head_type in ['action_type', 'queued', 'delay', 'selected_units', 'target_unit', 'target_location']:
        ent = -target_policy_probs_dict[head_type] * target_policy_log_probs_dict[head_type]
        if head_type == 'selected_units':
            ent = ent.sum(dim=-1) / (
                EPS + torch.log(mask['selected_units_logits_mask'].float().sum(dim=-1) + 1).unsqueeze(-1)
            )  # normalize
            ent = (ent * mask['selected_units_mask']).sum(-1)
            ent = ent.div(mask['selected_units_mask'].sum(-1) + EPS)
        elif head_type == 'target_unit':
            # normalize by unit
            ent = ent.sum(dim=-1) / (EPS + torch.log(mask['target_units_logits_mask'].float().sum(dim=-1) + 1))
        else:
            ent = ent.sum(dim=-1) / torch.log(torch.FloatTensor([ent.shape[-1]]).to(ent.device))
        if head_type not in ['action_type', 'delay']:
            ent = ent * mask['actions_mask'][head_type]
        entropy = ent.mean()
        entropy_dict['entropy/' + head_type] = entropy.item()
        total_entropy_loss += (-entropy * head_weights_dict[head_type])
    return total_entropy_loss, entropy_dict


def kl_error(
    target_policy_log_probs_dict, teacher_policy_logits_dict, mask, game_steps, action_type_kl_steps, head_weights_dict
):
    total_kl_loss = 0.
    kl_loss_dict = {}

    for head_type in ['action_type', 'queued', 'delay', 'selected_units', 'target_unit', 'target_location']:
        target_policy_log_probs = target_policy_log_probs_dict[head_type]
        teacher_policy_logits = teacher_policy_logits_dict[head_type]

        teacher_policy_log_probs = F.log_softmax(teacher_policy_logits, dim=-1)
        teacher_policy_probs = torch.exp(teacher_policy_log_probs)
        kl = teacher_policy_probs * (teacher_policy_log_probs - target_policy_log_probs)

        kl = kl.sum(dim=-1)
        if head_type == 'selected_units':
            kl = (kl * mask['selected_units_mask']).sum(-1)
        if head_type not in ['action_type', 'delay']:
            kl = kl * mask['actions_mask'][head_type]
        if head_type == 'action_type':
            flag = game_steps < action_type_kl_steps
            action_type_kl = kl * flag
            action_type_kl_loss = action_type_kl.mean()
            kl_loss_dict['kl/extra_at'] = action_type_kl_loss.item()
        kl_loss = kl.mean()
        total_kl_loss += (kl_loss * head_weights_dict[head_type])
        kl_loss_dict['kl/' + head_type] = kl_loss.item()
    return total_kl_loss, action_type_kl_loss, kl_loss_dict


class DIStarPolicy(Policy):
    config = dict(
        type='distar',
        on_policy=False,
        cuda=True,
        learning_rate=1e-5,
        model=dict(),
        learn=dict(multi_gpu=False, ),
        loss_weights=dict(
            baseline=dict(
                winloss=10.0,
                build_order=0.0,
                built_unit=0.0,
                effect=0.0,
                upgrade=0.0,
                battle=0.0,
            ),
            vtrace=dict(
                winloss=1.0,
                build_order=0.0,
                built_unit=0.0,
                effect=0.0,
                upgrade=0.0,
                battle=0.0,
            ),
            upgo=dict(winloss=1.0, ),
            kl=0.02,
            action_type_kl=0.1,
            entropy=0.0001,
        ),
        vtrace_head_weights=dict(
            action_type=1.0,
            delay=1.0,
            queued=1.0,
            select_unit_num_logits=1.0,
            selected_units=0.01,
            target_unit=1.0,
            target_location=1.0,
        ),
        upgo_head_weights=dict(
            action_type=1.0,
            delay=1.0,
            queued=1.0,
            select_unit_num_logits=1.0,
            selected_units=0.01,
            target_unit=1.0,
            target_location=1.0,
        ),
        entropy_head_weights=dict(
            action_type=1.0,
            delay=1.0,
            queued=1.0,
            select_unit_num_logits=1.0,
            selected_units=0.01,
            target_unit=1.0,
            target_location=1.0,
        ),
        kl_head_weights=dict(
            action_type=1.0,
            delay=1.0,
            queued=1.0,
            select_unit_num_logits=1.0,
            selected_units=0.01,
            target_unit=1.0,
            target_location=1.0,
        ),
        kl=dict(action_type_kl_steps=2400, ),
        gammas=dict(
            baseline=dict(
                winloss=1.0,
                build_order=1.0,
                built_unit=1.0,
                effect=1.0,
                upgrade=1.0,
                battle=0.997,
            ),
            pg=dict(
                winloss=1.0,
                build_order=1.0,
                built_unit=1.0,
                effect=1.0,
                upgrade=1.0,
                battle=0.997,
            ),
        ),
        grad_clip=dict(threshold=1.0, )
    )

    def _create_model(
            self,
            cfg: EasyDict,
            model: Optional[torch.nn.Module] = None,
            enable_field: Optional[List[str]] = None
    ) -> torch.nn.Module:
        assert model is None, "not implemented user-defined model"
        assert len(enable_field) == 1, "only support distributed enable policy"
        field = enable_field[0]
        if field == 'learn':
            return Model(self._cfg.model, use_value_network=True)
        else:
            raise KeyError("invalid policy mode: {}".format(field))

    def _init_learn(self):
        self.learn_model = model_wrap(self._model, 'base')
        self.head_types = ['action_type', 'delay', 'queued', 'target_unit', 'selected_units', 'target_location']
        # policy parameters
        self.gammas = self._cfg.gammas
        self.loss_weights = self._cfg.loss_weights
        self.action_type_kl_steps = self._cfg.kl.action_type_kl_steps
        self.vtrace_head_weights = self._cfg.vtrace_head_weights
        self.upgo_head_weights = self._cfg.upgo_head_weights
        self.entropy_head_weights = self._cfg.entropy_head_weights
        self.kl_head_weights = self._cfg.kl_head_weights

        # optimizer
        self.optimizer = Adam(
            self.learn_model.parameters(),
            lr=self._cfg.learning_rate,
            betas=(0, 0.99),
            eps=1e-5,
        )
        # utils
        self.timer = EasyTimer(cuda=self._cfg.cuda)

    def _forward_learn(self, inputs: Dict):
        # ===========
        # pre-process
        # ===========
        inputs = collate_fn_learn(inputs)
        if self._cfg.cuda:
            inputs = to_device(inputs, self._device)

        # =============
        # model forward
        # =============
        # create loss show dict
        loss_info_dict = {}
        with self.timer:
            model_output = self.learn_model.rl_learn_forward(**inputs)
        loss_info_dict['model_forward_time'] = self.timer.value

        # ===========
        # preparation
        # ===========
        target_policy_logits_dict = model_output['target_logit']  # shape (T,B)
        baseline_values_dict = model_output['value']  # shape (T+1,B)
        behaviour_action_log_probs_dict = model_output['action_log_prob']  # shape (T,B)
        teacher_policy_logits_dict = model_output['teacher_logit']  # shape (T,B)
        masks_dict = model_output['mask']  # shape (T,B)
        actions_dict = model_output['action']  # shape (T,B)
        rewards_dict = model_output['reward']  # shape (T,B)
        game_steps = model_output['step']  # shape (T,B) target_action_log_prob

        flag = rewards_dict['winloss'][-1] == 0
        for filed in baseline_values_dict.keys():
            baseline_values_dict[filed][-1] *= flag

        # create preparation info dict
        target_policy_probs_dict = {}
        target_policy_log_probs_dict = {}
        target_action_log_probs_dict = {}
        clipped_rhos_dict = {}

        # ============================================================
        # get distribution info for behaviour policy and target policy
        # ============================================================
        for head_type in self.head_types:
            # take info from correspondent input dict
            target_policy_logits = target_policy_logits_dict[head_type]
            actions = actions_dict[head_type]
            # compute target log_probs, probs(for entropy,kl), target_action_log_probs, log_rhos(for pg_loss, upgo_loss)
            pi_target = torch.distributions.Categorical(logits=target_policy_logits)
            target_policy_probs = pi_target.probs
            target_policy_log_probs = pi_target.logits
            target_action_log_probs = pi_target.log_prob(actions)
            behaviour_action_log_probs = behaviour_action_log_probs_dict[head_type]

            with torch.no_grad():
                log_rhos = target_action_log_probs - behaviour_action_log_probs
                if head_type == 'selected_units':
                    log_rhos *= masks_dict['selected_units_mask']
                    log_rhos = log_rhos.sum(dim=-1)
                rhos = torch.exp(log_rhos)
                clipped_rhos = rhos.clamp_(max=1)
            # save preparation results to correspondent dict
            target_policy_probs_dict[head_type] = target_policy_probs
            target_policy_log_probs_dict[head_type] = target_policy_log_probs
            if head_type == 'selected_units':
                target_action_log_probs.masked_fill_(~masks_dict['selected_units_mask'], 0)
                target_action_log_probs = target_action_log_probs.sum(-1)
            target_action_log_probs_dict[head_type] = target_action_log_probs
            # log_rhos_dict[head_type] = log_rhos
            clipped_rhos_dict[head_type] = clipped_rhos

        # ====================
        # vtrace loss
        # ====================
        total_vtrace_loss = 0.
        vtrace_loss_dict = {}

        for field, baseline in baseline_values_dict.items():
            baseline_value = baseline_values_dict[field]
            reward = rewards_dict[field]
            for head_type in self.head_types:
                weight = self.vtrace_head_weights[head_type]
                if head_type not in ['action_type', 'delay']:
                    weight = weight * masks_dict['actions_mask'][head_type]
                # if field in ['build_order', 'built_unit', 'effect']:
                #    weight = weight * masks_dict[field + '_mask']

                data_item = vtrace_data_with_rho(
                    target_action_log_probs_dict[head_type], clipped_rhos_dict[head_type], baseline_value, reward,
                    weight
                )
                vtrace_loss_item = vtrace_error_with_rho(data_item, gamma=1.0, lambda_=1.0)

                vtrace_loss_dict['vtrace/' + field + '/' + head_type] = vtrace_loss_item.item()
                total_vtrace_loss += self.loss_weights.vtrace[field] * self.vtrace_head_weights[head_type
                                                                                                ] * vtrace_loss_item

        loss_info_dict.update(vtrace_loss_dict)

        # ===========
        # upgo loss
        # ===========
        upgo_loss_dict = {}
        total_upgo_loss = 0.
        for head_type in self.head_types:
            weight = self.upgo_head_weights[head_type]
            if head_type not in ['action_type', 'delay']:
                weight = weight * masks_dict['actions_mask'][head_type]

            data_item = upgo_data(
                target_action_log_probs_dict[head_type], clipped_rhos_dict[head_type], baseline_values_dict['winloss'],
                rewards_dict['winloss'], weight
            )
            upgo_loss_item = upgo_error(data_item)

            total_upgo_loss += upgo_loss_item
            upgo_loss_dict['upgo/' + head_type] = upgo_loss_item.item()
        total_upgo_loss *= self.loss_weights.upgo.winloss
        loss_info_dict.update(upgo_loss_dict)

        # ===========
        # critic loss
        # ===========
        total_critic_loss = 0.
        # field is from ['winloss', 'build_order', 'built_unit', 'effect', 'upgrade', 'battle']
        for field, baseline in baseline_values_dict.items():
            reward = rewards_dict[field]
            # Notice: in general, we need to include done when we consider discount factor, but in our implementation
            # of alphastar, traj_data(with size equal to unroll-len) sent from actor comes from the same episode.
            # If the game is draw, we don't consider it is actually done
            # if field in ['build_order', 'built_unit', 'effect']:
            #    weight = masks_dict[[field + '_mask']]
            # else:
            #    weight = None
            weight = None

            field_data = td_lambda_data(baseline, reward, weight)
            critic_loss = td_lambda_error(field_data, gamma=self.gammas.baseline[field])

            total_critic_loss += self.loss_weights.baseline[field] * critic_loss
            loss_info_dict['td/' + field] = critic_loss.item()
            loss_info_dict['reward/' + field] = reward.float().mean().item()
            loss_info_dict['value/' + field] = baseline.mean().item()
        loss_info_dict['reward/battle'] = rewards_dict['battle'].float().mean().item()

        # ============
        # entropy loss
        # ============
        total_entropy_loss, entropy_dict = \
            entropy_error(target_policy_probs_dict, target_policy_log_probs_dict, masks_dict,
                          head_weights_dict=self.entropy_head_weights)

        total_entropy_loss *= self.loss_weights.entropy
        loss_info_dict.update(entropy_dict)

        # =======
        # kl loss
        # =======
        total_kl_loss, action_type_kl_loss, kl_loss_dict = \
            kl_error(target_policy_log_probs_dict, teacher_policy_logits_dict, masks_dict, game_steps,
                     action_type_kl_steps=self.action_type_kl_steps, head_weights_dict=self.kl_head_weights)
        total_kl_loss *= self.loss_weights.kl
        action_type_kl_loss *= self.loss_weights.action_type_kl
        loss_info_dict.update(kl_loss_dict)

        # ======
        # update
        # ======
        total_loss = (
            total_vtrace_loss + total_upgo_loss + total_critic_loss + total_entropy_loss + total_kl_loss +
            action_type_kl_loss
        )
        with self.timer:
            self.optimizer.zero_grad()
            total_loss.backward()
            if self._cfg.learn.multi_gpu:
                self.sync_gradients()
            gradient = torch.nn.utils.clip_grad_norm_(self.learn_model.parameters(), self._cfg.grad_clip.threshold, 2)
            self.optimizer.step()

        loss_info_dict['backward_time'] = self.timer.value
        loss_info_dict['total_loss'] = total_loss
        loss_info_dict['gradient'] = gradient
        return loss_info_dict

    def _monitor_var_learn(self):
        ret = ['total_loss', 'kl/extra_at', 'gradient', 'backward_time', 'model_forward_time']
        for k1 in ['winloss', 'build_order', 'built_unit', 'effect', 'upgrade', 'battle', 'upgo', 'kl', 'entropy']:
            for k2 in ['reward', 'value', 'td', 'action_type', 'delay', 'queued', 'selected_units', 'target_unit',
                       'target_location']:
                ret.append(k1 + '/' + k2)
        return ret

    def _state_dict(self) -> Dict:
        return {
            'model': self.learn_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, _state_dict: Dict) -> None:
        self.learn_model.load_state_dict(_state_dict['model'])
        self.optimizer.load_state_dict(_state_dict['optimizer'])

    def _init_collect(self):
        pass

    def _forward_collect(self, data):
        pass

    def _process_transition(self):
        pass

    def _get_train_sample(self):
        pass

    _init_eval = _init_collect
    _forward_eval = _forward_collect
