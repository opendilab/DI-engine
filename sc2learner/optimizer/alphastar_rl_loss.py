"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. base class for supervised learning on linklink, including basic processes.
"""
import collections
from collections import namedtuple, OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from sc2learner.optimizer.base_loss import BaseLoss
from sc2learner.torch_utils import levenshtein_distance, hamming_distance, same_shape
from sc2learner.rl_utils import td_lambda_loss, vtrace_loss, upgo_loss, compute_importance_weights, entropy
from sc2learner.utils import list_dict2dict_list, get_rank
from sc2learner.data import diff_shape_collate


def build_temperature_scheduler(temperature):
    """
        Overview: use config to initialize scheduler. Use naive temperature scheduler as default.
        Arguments:
            - cfg (:obj:`dict`): scheduler config
        Returns:
            - (:obj`Scheduler`): scheduler created by this function
    """
    class ConstantTemperatureSchedule:
        def __init__(self, init_val=0.1):
            self.init_val = init_val

        def step(self):
            return self.init_val

        def value(self):
            return self.init_val

    return ConstantTemperatureSchedule(init_val=temperature)


def pad_stack(data, pad_val):
    assert (isinstance(data, list))
    dtype, device = data[0].dtype, data[0].device
    shapes = torch.LongTensor([t.shape for t in data])
    max_dims = shapes.max(dim=0)[0].tolist()
    size = [len(data)] + max_dims
    new_data = torch.full(size, pad_val).to(dtype=dtype, device=device)
    for idx, d in enumerate(data):
        slices = [slice(0, n) for n in d.shape]
        new_data[idx][slices] = d
    return new_data


class AlphaStarRLLoss(BaseLoss):
    def __init__(self, agent, train_config, model_config):
        self.action_keys = ['action_type', 'delay', 'queued', 'selected_units', 'target_units', 'target_location']
        self.loss_keys = ['td_lambda', 'vtrace', 'upgo', 'kl', 'action_type_kl', 'entropy']
        self.rollout_outputs = namedtuple(
            "rollout_outputs", [
                'target_outputs', 'behaviour_outputs', 'teacher_outputs', 'baselines', 'rewards', 'target_actions',
                'behaviour_actions', 'teacher_actions', 'game_seconds'
            ]
        )
        self.agent = agent
        self.T = train_config.trajectory_len
        self.batch_size = train_config.batch_size
        self.vtrace_rhos_min_clip = train_config.vtrace.min_clip
        self.upgo_rhos_min_clip = train_config.upgo.min_clip
        self.action_output_types = train_config.action_output_types
        assert (all([t in ['value', 'logit'] for t in self.action_output_types.values()]))
        self.action_type_kl_seconds = train_config.kl.action_type_kl_seconds
        self.use_target_state = train_config.use_target_state

        self.loss_weights = train_config.loss_weights
        self.temperature_scheduler = build_temperature_scheduler(
            train_config.temperature
        )  # get naive temperature scheduler

        self.dtype = torch.float
        self.rank = get_rank()
        self.device = 'cuda:{}'.format(self.rank % 8) if train_config.use_cuda and torch.cuda.is_available() else 'cpu'
        self.pad_value = -1e6

    def register_log(self, variable_record, tb_logger):
        for k in self.loss_keys:
            variable_record.register_var(k + '_loss')
            tb_logger.register_var(k + '_loss')

    def compute_loss(self, data):
        """
            Overview: Overwrite function, main process of training
            Arguments:
                - data (:obj:`batch_data`): batch_data created by dataloader

        """
        rollout_outputs = self._rollout(data)
        # TODO(nyz) apply the importance sampling weight in gradient update
        target_outputs, behaviour_outputs, teacher_outputs, baselines = rollout_outputs[:4]
        rewards, target_actions, behaviour_actions, teacher_actions, game_seconds = rollout_outputs[4:]

        # td_lambda and v_trace
        actor_critic_loss = 0.
        td_lambda_loss_val = {}
        vtrace_loss_val = {}
        for field, baseline in baselines.items():
            reward = rewards[field]
            td_lambda_loss = self._td_lambda_loss(baseline, reward) * self.loss_weights.baseline[field]
            td_lambda_loss_val[field] = td_lambda_loss.item()
            vtrace_loss = self._vtrace_pg_loss(
                baseline, reward, target_outputs, behaviour_outputs, target_actions, behaviour_actions
            ) * self.loss_weights.pg[field]
            vtrace_loss_val[field] = vtrace_loss.item()
            actor_critic_loss += td_lambda_loss + vtrace_loss
        # upgo loss
        upgo_loss = self._upgo_loss(
            baselines['winloss'], rewards['winloss'], target_outputs, behaviour_outputs, target_actions,
            behaviour_actions
        ) * self.loss_weights.upgo['winloss']

        # human kl loss
        kl_loss, action_type_kl_loss = self._human_kl_loss(
            target_outputs, teacher_outputs, target_actions, teacher_actions, game_seconds
        )
        kl_loss *= self.loss_weights.kl
        action_type_kl_loss *= self.loss_weights.action_type_kl
        # entropy loss
        ent_loss = self._entropy_loss(target_outputs) * self.loss_weights.entropy

        total_loss = actor_critic_loss + kl_loss + action_type_kl_loss + ent_loss + upgo_loss
        return {
            'total_loss': total_loss,
            'kl_loss': kl_loss,
            'action_type_kl_loss': action_type_kl_loss,
            'ent_loss': ent_loss,
            'upgo_loss': upgo_loss,
            'td_lambda_loss': sum(td_lambda_loss_val.values()),
            'vtrace_loss': sum(vtrace_loss_val.values()),
        }

    def _rollout(self, data):
        temperature = self.temperature_scheduler.step()
        next_state_home, next_state_away = None, None
        outputs_dict = OrderedDict({k: [] for k in self.rollout_outputs._fields})
        for idx, step_data in enumerate(data):
            if self.use_target_state and next_state_home is not None:
                step_data['home']['prev_state'] = next_state_home
                step_data['away']['prev_state'] = next_state_away
            target_actions, target_outputs, baselines, next_state_home, next_state_away = (
                self.agent.compute_action_value(step_data, temperature)
            )
            target_actions.pop('action_entity_raw')
            target_actions = {k: diff_shape_collate(v) for k, v in target_actions.items()}
            # add to outputs
            home = step_data['home']
            outputs_dict['target_outputs'].append(target_outputs)
            outputs_dict['behaviour_outputs'].append(home['behaviour_outputs'])
            outputs_dict['teacher_outputs'].append(home['teacher_outputs'])
            outputs_dict['baselines'].append(baselines)
            outputs_dict['rewards'].append(home['rewards'])
            outputs_dict['target_actions'].append(target_actions)
            outputs_dict['behaviour_actions'].append(home['actions'])
            outputs_dict['teacher_actions'].append(home['teacher_actions'])
        # last baselines/values
        last_obs = {'home': data[-1]['home_next'], 'away': data[-1]['away_next']}
        if self.use_target_state and next_state_home is not None:
            last_obs['home']['prev_state'] = next_state_home
            last_obs['away']['prev_state'] = next_state_away
        last_baselines = self.agent.compute_action_value(last_obs, temperature).baselines
        outputs_dict['baselines'].append(last_baselines)
        # change dim(tra_len, key, bs->key, tra_len, bs)
        for k in outputs_dict.keys():
            if k != 'game_seconds':
                outputs_dict[k] = list_dict2dict_list(outputs_dict[k])
        outputs_dict['baselines'] = {
            k: torch.stack(v, dim=0)
            for k, v in zip(outputs_dict['baselines']._fields, outputs_dict['baselines']) if v[0] is not None
        }
        # each value: (batch_size, 1) -> stack+squeeze -> (tra_len, batch_size)
        outputs_dict['rewards'] = {k: torch.stack(v, dim=0).squeeze(2) for k, v in outputs_dict['rewards'].items()}
        # add game_seconds
        outputs_dict['game_seconds'].extend(data[-1]['home']['game_seconds'])
        return self.rollout_outputs(*outputs_dict.values())  # outputs_dict is a OrderedDict

    def _td_lambda_loss(self, baseline, reward):
        """
            default: gamma=1.0, lamda=0.8
        """
        assert (isinstance(baseline, torch.Tensor) and baseline.shape[0] == self.T + 1)
        assert (isinstance(reward, torch.Tensor) and reward.shape[0] == self.T)
        return td_lambda_loss(baseline, reward)

    def _vtrace_pg_loss(self, baseline, reward, target_outputs, behaviour_outputs, target_actions, behaviour_actions):
        """
            seperated vtrace loss
        """
        def _vtrace(target_output, behaviour_output, action, action_output_type):
            clipped_rhos = compute_importance_weights(
                target_output,
                behaviour_output,
                action_output_type,
                action,
                min_clip=self.vtrace_rhos_min_clip,
                device=self.device
            )
            clipped_cs = clipped_rhos
            return vtrace_loss(target_output, action_output_type, clipped_rhos, clipped_cs, action, reward, baseline)

        target_outputs, behaviour_outputs, target_actions, behaviour_actions = self._filter_pack(
            target_outputs, behaviour_outputs, target_actions, behaviour_actions
        )
        loss = 0.
        for k in self.action_keys:
            if len(behaviour_actions[k]) > 0:
                loss += _vtrace(
                    target_outputs[k], behaviour_outputs[k], behaviour_actions[k], self.action_output_types[k]
                )

        return loss

    def _upgo_loss(self, baseline, reward, target_outputs, behaviour_outputs, target_actions, behaviour_actions):
        def _upgo(target_output, behaviour_output, action, action_output_type):
            clipped_rhos = compute_importance_weights(
                target_output,
                behaviour_output,
                action_output_type,
                action,
                min_clip=self.upgo_rhos_min_clip,
                device=self.device
            )
            return upgo_loss(target_output, action_output_type, clipped_rhos, action, reward, baseline)

        target_outputs, behaviour_outputs, target_actions, behaviour_actions = self._filter_pack(
            target_outputs, behaviour_outputs, target_actions, behaviour_actions
        )
        loss = 0.
        for k in self.action_keys:
            if len(behaviour_actions[k]) > 0:
                loss += _upgo(
                    target_outputs[k], behaviour_outputs[k], behaviour_actions[k], self.action_output_types[k]
                )
        return loss

    def _human_kl_loss(self, target_outputs, teacher_outputs, target_actions, teacher_actions, game_seconds):
        def kl(stu, tea):
            stu = F.log_softmax(stu, dim=-1)
            tea = F.softmax(tea, dim=-1)
            return F.kl_div(stu, tea)

        target_outputs, teacher_outputs, target_actions, teacher_actions = self._filter_pack(
            target_outputs, teacher_outputs, target_actions, teacher_actions
        )
        kl_loss = 0.
        for k in self.action_keys:
            if len(teacher_outputs[k]) > 0:
                if self.action_output_types[k] == 'logit':
                    if k == 'selected_units':
                        for t in range(self.T):
                            for b in range(self.batch_size):
                                if teacher_outputs[k][t][b] is not None:
                                    kl_loss += kl(target_outputs[k][t][b], teacher_outputs[k][t][b])
                    else:
                        kl_loss += kl(target_outputs[k], teacher_outputs[k])
                elif self.action_output_types[k] == 'value':
                    target_output, teacher_output = target_outputs[k], teacher_outputs[k]
                    kl_loss += F.l1_loss(target_output, teacher_output)

        action_type_kl_loss = torch.zeros(1).to(dtype=self.dtype, device=self.device)
        for i in range(len(game_seconds)):
            if game_seconds[i] < self.action_type_kl_seconds:
                # batch dim
                action_type_kl_loss += kl(target_outputs['action_type'][:, i], teacher_outputs['action_type'][:, i])

        return kl_loss, action_type_kl_loss

    def _entropy_loss(self, target_outputs):
        loss = torch.zeros(1).to(dtype=self.dtype, device=self.device)
        target_outputs = self._filter_pack_valid_outputs(target_outputs)
        for k in self.action_keys:
            if self.action_output_types[k] == 'logit':
                if len(target_outputs[k]) > 0:
                    loss += entropy(target_outputs[k])
        return loss

    def _filter_pack(self, pred_outputs, base_outputs, pred_actions, base_actions):
        """
            Overview: According to base_actions to filter pred_actions, and pack all actions and outputs
        """
        new_pred_outputs = {}
        new_base_outputs = {}
        new_pred_actions = {}
        new_base_actions = {}
        for k in ['action_type', 'delay']:
            # T, B, 1
            new_base_actions[k] = torch.stack(base_actions[k], dim=0)
            new_pred_actions[k] = torch.stack(pred_actions[k], dim=0)
            # T, B, N
            new_pred_outputs[k] = torch.stack(pred_outputs[k], dim=0)
            new_base_outputs[k] = torch.stack(base_outputs[k], dim=0)

        eq = torch.eq(new_base_actions['action_type'].squeeze(2), new_pred_actions['action_type'].squeeze(2))
        eq_idx = torch.nonzero(eq).tolist()

        # queued, selected_units, target_units, target_location
        for k in ['queued', 'selected_units', 'target_units', 'target_location']:
            for d, new_d in zip([pred_outputs, base_outputs, pred_actions, base_actions],
                                [new_pred_outputs, new_base_outputs, new_pred_actions, new_base_actions]):
                valid_list = []
                # action type equal
                for t_idx, b_idx in eq_idx:
                    tmp = d[k][t_idx][b_idx]
                    if tmp is not None:
                        valid_list.append(tmp)
                if len(valid_list) > 0:
                    if k == 'selected_units':
                        new_d[k] = [[None for _ in range(self.batch_size)] for _ in range(self.T)]
                    else:
                        # (T, B, n_output_dim)
                        if not same_shape(valid_list):
                            valid_list = pad_stack(valid_list, self.pad_value)
                        ref = valid_list[0]
                        size = (self.T, self.batch_size) + ref.shape
                        new_d[k] = torch.zeros(size).to(dtype=ref.dtype, device=ref.device)
                    for item, (t_idx, b_idx) in zip(valid_list, eq_idx):
                        new_d[k][t_idx][b_idx] = item
                else:
                    new_d[k] = []
        # cut off selected_units num by base
        for t in range(self.T):
            for b in range(self.batch_size):
                if len(new_base_actions['selected_units']) <= 0:
                    continue
                if new_base_actions['selected_units'][t][b] is not None:
                    base_num = new_base_outputs['selected_units'][t][b].shape[0]
                    pred_num = new_pred_outputs['selected_units'][t][b].shape[0]
                    if base_num == pred_num:
                        continue
                    elif base_num < pred_num:
                        new_pred_actions['selected_units'][t][b] = new_pred_actions['selected_units'][t][b][:base_num]
                        new_pred_outputs['selected_units'][t][b] = new_pred_outputs['selected_units'][t][b][:base_num]
                    else:
                        new_base_actions['selected_units'][t][b] = new_base_actions['selected_units'][t][b][:pred_num]
                        new_base_outputs['selected_units'][t][b] = new_base_outputs['selected_units'][t][b][:pred_num]

        return new_pred_outputs, new_base_outputs, new_pred_actions, new_base_actions

    def _filter_pack_valid_outputs(self, outputs):
        """
            Overview: select outputs which is not None and pack them
        """
        new_outputs = {}
        for k in ['action_type', 'delay']:
            new_outputs[k] = torch.stack(outputs[k], dim=0)
        for k in ['queued', 'selected_units', 'target_units', 'target_location']:
            valid_list = []
            for t in range(self.T):
                for b in range(self.batch_size):
                    if outputs[k][t][b] is not None:
                        valid_list.append(outputs[k][t][b])
            if len(valid_list) > 0:
                if same_shape(valid_list):
                    new_outputs[k] = torch.stack(valid_list, dim=0)
                else:
                    new_outputs[k] = pad_stack(valid_list, self.pad_value)
            else:
                new_outputs[k] = []
        return new_outputs
