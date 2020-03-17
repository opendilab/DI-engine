"""
Copyright 2020 Sensetime X-lab. All Rights Reserved
"""

from collections import defaultdict

import torch
from torch.nn import functional as F

from pysc2.lib.static_data import ACTIONS_REORDER_INV


def delay_l1(p, l):
    l = l.float()  # noqa
    p = p.float()
    base = -1.73e-5 * l ** 3 + 1.89e-3 * l ** 2 - 5.8e-2 * l + 0.61
    loss = torch.abs(p - l) - base * l
    return loss.clamp(0).mean().item()


def accuracy(p, l):
    return p.eq(l).sum().item()


def IOU(p, l):
    p_set = set(p.tolist())
    l_set = set(l.tolist())
    union = p_set.union(l_set)
    intersect = p_set.intersection(l_set)
    return len(intersect) * 1.0 / len(union)


def L2(p, l):
    return F.mse_loss(p.float(), l.float()).item()


class SupervisedCriterion:
    """
        Overview: AlphaStar supervised learning evaluate criterion
        Interface: __init__, update, get_stat, to_string
    """

    def __init__(self):
        self.action_heads = ['delay', 'queued', 'selected_units', 'target_units', 'target_location']

        self.action_head_criterion_mapping = dict(
            delay=delay_l1,
            queued=accuracy,
            selected_units=IOU,
            target_units=accuracy,
            target_location=L2
        )

        self.buffer = {head_name: [] for head_name in self.action_heads}

        self.action_type = defaultdict(list)
        self.action_type_hard_case = defaultdict(int)

    def update(self, pred, target):
        """
            Overview: update eval criterion one step
            Arguments:
                - pred (:obj:`dict`): predict action
                - target (:obj:`target`) ground truth action
            Others:
                - action type: accuracy
                - delay: delay L1 distance (only for matched action type)
                - queued: accuracy (only for matched action type and the action with this arribute)
                - selected units: IOU (only for matched action type and the action with this arribute)
                - target units: accuracy (only for matched action type and the action with this arribute)
                - target location: L2 distance (only for matched action type and the action with this arribute)
        """
        with torch.no_grad():
            pred_action_type = pred['action_type'][0]
            target_action_type = target['action_type'][0]
            action_type = target_action_type.item()
            if pred_action_type == target_action_type:
                for k in self.action_heads:
                    if isinstance(pred[k][0], torch.Tensor):
                        criterion = self.action_head_criterion_mapping[k](pred[k][0], target[k][0])
                        self.buffer[k].append([criterion, target_action_type.item()])
                self.action_type[action_type].append(1)
            else:
                self.action_type[action_type].append(0)
                self.action_type_hard_case[action_type] += 1

    def get_stat(self):
        avg = {}
        hard_case = {}
        threshold = {k: v for k, v in zip(self.action_heads, [0.8, 0.95, 0.33, 0.8, 2.9])}
        for head_name in self.action_heads:
            attr = self.buffer[head_name]
            criterion_dict = defaultdict(list)
            hard_case_dict = defaultdict(int)
            th = threshold[head_name]
            for t in attr:
                criterion_dict[t[1]].append(t[0])
                if head_name == 'target_location':
                    if t[0] > th:
                        hard_case_dict[t[1]] += 1
                else:
                    if t[0] < th:
                        hard_case_dict[t[1]] += 1

            avg[head_name] = sorted([[k, sum(v) / (len(v) + 1e-8)] for k, v in criterion_dict.items()],
                                    key=lambda x: x[1])
            if len(avg[head_name]) > 0:
                val = list(zip(*avg[head_name]))[1]
                avg[head_name].insert(0, ['total', sum(val) / (len(val) + 1e-8)])
            hard_case[head_name] = sorted([[k, v] for k, v in hard_case_dict.items()], key=lambda x: x[1], reverse=True)

        action_type = sorted([[k, sum(v) / (len(v) + 1e-8)] for k, v in self.action_type.items()], key=lambda x: x[1])
        val = list(zip(*action_type))[1]
        action_type.insert(0, ['total', sum(val) / (len(val) + 1e-8)])
        hard_case_action_type = sorted([[k, v] for k, v in self.action_type_hard_case.items()],
                                       key=lambda x: x[1], reverse=True)
        ret = {
            'action_type acc': action_type,
            'delay l1': avg['delay'],
            'queued acc': avg['queued'],
            'selected_units IOU': avg['selected_units'],
            'target_units acc': avg['target_units'],
            'target_location L2': avg['target_location'],

            'action_type hard case': hard_case_action_type,
            'delay hard case': hard_case['delay'],
            'queued hard case': hard_case['queued'],
            'selected_units hard case': hard_case['selected_units'],
            'target_units hard case': hard_case['target_units'],
            'target_location hard case': hard_case['target_location']
        }
        return to_string(ret)


def to_string(data):
    s = "\n"
    for k, v in data.items():
        if 'hard case' in k:
            v_str = ['{}({})'.format(ACTIONS_REORDER_INV[name], count) for name, count in v]
            s += '{}: {}\n'.format(k, '\t'.join(v_str))
        else:
            v_str = ['{}({:.4f})'.format(name, val) for name, val in v]
            s += '{}: {}\n'.format(k, '\t'.join(v_str))
    return s
