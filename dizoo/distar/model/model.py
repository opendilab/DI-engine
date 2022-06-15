from collections import OrderedDict
from typing import Dict, Tuple, List
from torch import Tensor

import os.path as osp
import torch
import torch.nn as nn

from ding.utils import read_yaml_config, deep_merge_dicts
from ding.utils.data import default_collate
from ding.torch_utils import detach_grad, script_lstm
from dizoo.distar.envs import MAX_SELECTED_UNITS_NUM
from .encoder import Encoder
from .obs_encoder.value_encoder import ValueEncoder
from .policy import Policy
from .value import ValueBaseline

alphastar_model_default_config = read_yaml_config(osp.join(osp.dirname(__file__), "actor_critic_default_config.yaml"))


class Model(nn.Module):

    def __init__(self, cfg, use_value_network=False):
        super(Model, self).__init__()
        self.cfg = deep_merge_dicts(alphastar_model_default_config, cfg).model
        self.encoder = Encoder(self.cfg)
        self.policy = Policy(self.cfg)
        self.use_value_network = use_value_network
        if self.use_value_network:
            self.use_value_feature = self.cfg.value.use_value_feature
            if self.use_value_feature:
                self.value_encoder = ValueEncoder(self.cfg.value)
            self.value_networks = nn.ModuleDict()
            for k, v in self.cfg.value.items():
                if k in self.cfg.enable_baselines:
                    # creating a ValueBaseline network for each baseline, to be used in _critic_forward
                    value_cfg = v.param
                    value_cfg['use_value_feature'] = self.use_value_feature
                    self.value_networks[v.name] = ValueBaseline(value_cfg)
                    # name of needed cumulative stat items
        self.only_update_baseline = self.cfg.only_update_baseline
        self.core_lstm = script_lstm(
            self.cfg.encoder.core_lstm.input_size,
            self.cfg.encoder.core_lstm.hidden_size,
            self.cfg.encoder.core_lstm.num_layers,
            LN=True
        )

    def forward(
        self,
        spatial_info: Tensor,
        entity_info: Dict[str, Tensor],
        scalar_info: Dict[str, Tensor],
        entity_num: Tensor,
        hidden_state: List[Tuple[Tensor, Tensor]],
    ):
        lstm_input, scalar_context, baseline_feature, entity_embeddings, map_skip = \
            self.encoder(spatial_info, entity_info, scalar_info, entity_num)
        lstm_output, out_state = self.core_lstm(lstm_input.unsqueeze(dim=0), hidden_state)
        action_info, selected_units_num, logit, extra_units = self.policy(
            lstm_output.squeeze(dim=0), entity_embeddings, map_skip, scalar_context, entity_num
        )
        return action_info, selected_units_num, out_state

    def compute_logp_action(
        self, spatial_info: Tensor, entity_info: Dict[str, Tensor], scalar_info: Dict[str, Tensor], entity_num: Tensor,
        hidden_state: List[Tuple[Tensor, Tensor]], **kwargs
    ):
        lstm_input, scalar_context, baseline_feature, entity_embeddings, map_skip = \
            self.encoder(spatial_info, entity_info, scalar_info, entity_num)
        lstm_output, out_state = self.core_lstm(lstm_input.unsqueeze(dim=0), hidden_state)
        action_info, selected_units_num, logit, extra_units = self.policy(
            lstm_output.squeeze(dim=0), entity_embeddings, map_skip, scalar_context, entity_num
        )
        log_action_probs = {}
        for k, action in action_info.items():
            dist = torch.distributions.Categorical(logits=logit[k])
            action_log_probs = dist.log_prob(action)
            log_action_probs[k] = action_log_probs
        return {
            'action_info': action_info,
            'action_logp': log_action_probs,
            'selected_units_num': selected_units_num,
            'entity_num': entity_num,
            'hidden_state': out_state,
            'logit': logit,
            'extra_units': extra_units
        }

    def compute_teacher_logit(
        self, spatial_info: Tensor, entity_info: Dict[str, Tensor], scalar_info: Dict[str, Tensor], entity_num: Tensor,
        hidden_state: List[Tuple[Tensor, Tensor]], selected_units_num, action_info: Dict[str, Tensor], **kwargs
    ):
        lstm_input, scalar_context, baseline_feature, entity_embeddings, map_skip = \
            self.encoder(spatial_info, entity_info, scalar_info, entity_num)
        lstm_output, out_state = self.core_lstm(lstm_input.unsqueeze(dim=0), hidden_state)
        action_info, selected_units_num, logit = self.policy.train_forward(
            lstm_output.squeeze(dim=0), entity_embeddings, map_skip, scalar_context, entity_num, action_info,
            selected_units_num
        )
        return {
            'logit': logit,
            'hidden_state': out_state,
            'entity_num': entity_num,
            'selected_units_num': selected_units_num
        }

    def rl_learn_forward(
        self, spatial_info, entity_info, scalar_info, entity_num, hidden_state, action_info, selected_units_num,
        behaviour_logp, teacher_logit, mask, reward, step, batch_size, unroll_len, **kwargs
    ):
        assert self.use_value_network
        flat_action_info = {}
        for k, val in action_info.items():
            flat_action_info[k] = torch.flatten(val, start_dim=0, end_dim=1)
        flat_selected_units_num = torch.flatten(selected_units_num, start_dim=0, end_dim=1)

        lstm_input, scalar_context, baseline_feature, entity_embeddings, map_skip = \
            self.encoder(spatial_info, entity_info, scalar_info, entity_num)
        hidden_size = hidden_state[0][0].shape[-1]

        hidden_state = [
            [hidden_state[i][j].view(-1, batch_size, hidden_size)[0, :, :] for j in range(len(hidden_state[i]))]
            for i in range(len(hidden_state))
        ]
        lstm_output, out_state = self.core_lstm(lstm_input.view(-1, batch_size, lstm_input.shape[-1]), hidden_state)
        lstm_output = lstm_output.view(-1, lstm_output.shape[-1])

        policy_lstm_input = lstm_output.squeeze(dim=0)[:-batch_size]
        policy_entity_embeddings = entity_embeddings[:-batch_size]
        policy_map_skip = [map[:-batch_size] for map in map_skip]
        policy_scalar_context = scalar_context[:-batch_size]
        policy_entity_num = entity_num[:-batch_size]
        _, _, logits = self.policy.train_forward(
            policy_lstm_input, policy_entity_embeddings, policy_map_skip, policy_scalar_context, policy_entity_num,
            flat_action_info, flat_selected_units_num
        )

        # logits['selected_units'] = logits['selected_units'].mean(dim=1)
        critic_input = lstm_output.squeeze(0)
        # add state info
        if self.only_update_baseline:
            critic_input = detach_grad(critic_input)
            baseline_feature = detach_grad(baseline_feature)
        if self.use_value_feature:
            value_feature = kwargs['value_feature']
            value_feature = self.value_encoder(value_feature)
            critic_input = torch.cat([critic_input, value_feature, baseline_feature], dim=1)
        baseline_values = {}
        for k, v in self.value_networks.items():
            baseline_values[k] = v(critic_input)
        for k, val in logits.items():
            logits[k] = val.view(unroll_len, batch_size, *val.shape[1:])
        for k, val in baseline_values.items():
            baseline_values[k] = val.view(unroll_len + 1, batch_size)
        outputs = {}
        outputs['unroll_len'] = unroll_len
        outputs['batch_size'] = batch_size
        outputs['selected_units_num'] = selected_units_num
        logits['selected_units'] = torch.nn.functional.pad(
            logits['selected_units'], (
                0,
                0,
                0,
                MAX_SELECTED_UNITS_NUM - logits['selected_units'].shape[2],
            ), 'constant', -1e9
        )
        outputs['target_logit'] = logits
        outputs['value'] = baseline_values
        outputs['action_log_prob'] = behaviour_logp
        outputs['teacher_logit'] = teacher_logit
        outputs['mask'] = mask
        outputs['action'] = action_info
        outputs['reward'] = reward
        outputs['step'] = step

        return outputs

    def sl_learn_forward(
        self, spatial_info, entity_info, scalar_info, entity_num, selected_units_num, traj_lens, hidden_state,
        action_info, **kwargs
    ):
        batch_size = len(traj_lens)
        lstm_input, scalar_context, baseline_feature, entity_embeddings, map_skip = \
            self.encoder(spatial_info, entity_info, scalar_info, entity_num)
        lstm_input = lstm_input.view(-1, lstm_input.shape[0] // batch_size, lstm_input.shape[-1]).permute(1, 0, 2)
        lstm_output, out_state = self.core_lstm(lstm_input, hidden_state)
        lstm_output = lstm_output.permute(1, 0, 2).contiguous().view(-1, lstm_output.shape[-1])
        action_info, selected_units_num, logits = self.policy.train_forward(
            lstm_output, entity_embeddings, map_skip, scalar_context, entity_num, action_info, selected_units_num
        )
        return logits, action_info, out_state
