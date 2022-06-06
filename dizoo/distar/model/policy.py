from typing import List, Dict, Optional
from torch import Tensor
import torch
import torch.nn as nn

from dizoo.distar.envs import SELECTED_UNITS_MASK
from .head import DelayHead, QueuedHead, SelectedUnitsHead, TargetUnitHead, LocationHead, ActionTypeHead


class Policy(nn.Module):

    def __init__(self, cfg):
        super(Policy, self).__init__()
        self.whole_cfg = cfg
        self.cfg = cfg.model.policy
        self.action_type_head = ActionTypeHead(self.whole_cfg)
        self.delay_head = DelayHead(self.whole_cfg)
        self.queued_head = QueuedHead(self.whole_cfg)
        self.selected_units_head = SelectedUnitsHead(self.whole_cfg)
        self.target_unit_head = TargetUnitHead(self.whole_cfg)
        self.location_head = LocationHead(self.whole_cfg)

    def forward(
        self, lstm_output: Tensor, entity_embeddings: Tensor, map_skip: List[Tensor], scalar_context: Tensor,
        entity_num: Tensor
    ):
        action = torch.jit.annotate(Dict[str, Tensor], {})
        logit = torch.jit.annotate(Dict[str, Tensor], {})

        # action type
        logit['action_type'], action['action_type'], embeddings = self.action_type_head(lstm_output, scalar_context)

        #action arg delay
        logit['delay'], action['delay'], embeddings = self.delay_head(embeddings)

        logit['queued'], action['queued'], embeddings = self.queued_head(embeddings)

        # selected_units_head cannot be compiled to onnx due to indice
        su_mask = SELECTED_UNITS_MASK[action['action_type']]
        logit['selected_units'], action['selected_units'
                                        ], embeddings, selected_units_num, extra_units = self.selected_units_head(
                                            embeddings, entity_embeddings, entity_num, None, None, su_mask
                                        )

        logit['target_unit'], action['target_unit'] = self.target_unit_head(embeddings, entity_embeddings, entity_num)

        logit['target_location'], action['target_location'] = self.location_head(embeddings, map_skip)

        return action, selected_units_num, logit, extra_units

    def train_forward(
        self, lstm_output, entity_embeddings, map_skip: List[Tensor], scalar_context, entity_num, action_info,
        selected_units_num
    ):
        action = torch.jit.annotate(Dict[str, Tensor], {})
        logit = torch.jit.annotate(Dict[str, Tensor], {})

        # action type
        logit['action_type'], action['action_type'], embeddings = self.action_type_head(
            lstm_output, scalar_context, action_info['action_type']
        )

        #action arg delay
        logit['delay'], action['delay'], embeddings = self.delay_head(embeddings, action_info['delay'])

        logit['queued'], action['queued'], embeddings = self.queued_head(embeddings, action_info['queued'])

        logit['selected_units'], action['selected_units'], embeddings, selected_units_num, _ = self.selected_units_head(
            embeddings, entity_embeddings, entity_num, selected_units_num, action_info['selected_units']
        )

        logit['target_unit'], action['target_unit'] = self.target_unit_head(
            embeddings, entity_embeddings, entity_num, action_info['target_unit']
        )

        logit['target_location'], action['target_location'] = self.location_head(
            embeddings, map_skip, action_info['target_location']
        )
        return action, selected_units_num, logit
