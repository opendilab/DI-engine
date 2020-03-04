'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. Alphastar implementation for supervised learning on linklink, including basic processes.
'''
import math
import collections
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sl_learner import SLLearner
from pysc2.lib.static_data import ACTIONS_REORDER_INV, ACTIONS
from sc2learner.nn_utils import MultiLogitsLoss, build_criterion
from sc2learner.utils import to_device


def build_temperature_scheduler(cfg):
    '''
        Overview: use config to initialize scheduler. Use naive temperature scheduler as default.
        Arguments:
            - cfg (:obj:`dict`): scheduler config
        Returns:
            - (:obj`Scheduler`): scheduler created by this function
    '''
    class NaiveT(object):
        def __init__(self, init_val=0.1):
            self.init_val = init_val

        def step(self):
            return self.init_val

    return NaiveT(init_val=cfg.train.temperature)


class AlphastarSLCriterion(object):
    '''
        Overview: Alphastar supervised learning evaluate criterion
        Interface: __init__, update, get_stat, to_string
    '''

    def __init__(self):
        self.action_arg = ['delay', 'queued', 'selected_units', 'target_units', 'target_location']
        for k in self.action_arg:
            setattr(self, k, [])

        def delay_l1(p, l):
            l = l.float()  # noqa
            p = p.float()
            base = -1.73e-5*l**3 + 1.89e-3*l**2 - 5.8e-2*l + 0.61
            loss = torch.abs(p - l) - base*l
            return loss.clamp(0).mean().item()

        def accuracy(p, l):
            return p.eq(l).sum().item()

        def IOU(p, l):
            p_set = set(p.tolist())
            l_set = set(l.tolist())
            union = p_set.union(l_set)
            intersect = p_set.intersection(l_set)
            return len(intersect)*1.0/len(union)

        def L2(p, l):
            return F.mse_loss(p.float(), l.float()).item()

        self.action_arg_criterion = {k: v for k, v in zip(self.action_arg, [delay_l1, accuracy, IOU, accuracy, L2])}
        self.action_type = defaultdict(list)
        self.action_type_hard_case = defaultdict(int)

    def update(self, pred, target):
        '''
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
        '''
        with torch.no_grad():
            pred_action_type = pred['action_type'][0]
            target_action_type = target['action_type'][0]
            action_type = target_action_type.item()
            if pred_action_type == target_action_type:
                for k in self.action_arg:
                    if isinstance(pred[k][0], torch.Tensor):
                        criterion = self.action_arg_criterion[k](pred[k][0], target[k][0])
                        handle = getattr(self, k)
                        handle.append([criterion, target_action_type.item()])
                self.action_type[action_type].append(1)
            else:
                self.action_type[action_type].append(0)
                self.action_type_hard_case[action_type] += 1

    def get_stat(self):
        avg = {}
        hard_case = {}
        threshold = {k: v for k, v in zip(self.action_arg, [0.8, 0.95, 0.33, 0.8, 2.9])}
        for arg in self.action_arg:
            attr = getattr(self, arg)
            criterion_dict = defaultdict(list)
            hard_case_dict = defaultdict(int)
            th = threshold[arg]
            for t in attr:
                criterion_dict[t[1]].append(t[0])
                if arg == 'target_location':
                    if t[0] > th:
                        hard_case_dict[t[1]] += 1
                else:
                    if t[0] < th:
                        hard_case_dict[t[1]] += 1

            avg[arg] = sorted([[k, sum(v) / (len(v) + 1e-8)] for k, v in criterion_dict.items()], key=lambda x: x[1])
            if len(avg[arg]) > 0:
                val = list(zip(*avg[arg]))[1]
                avg[arg].insert(0, ['total', sum(val) / (len(val) + 1e-8)])
            hard_case[arg] = sorted([[k, v] for k, v in hard_case_dict.items()], key=lambda x: x[1], reverse=True)

        action_type = sorted([[k, sum(v) / (len(v) + 1e-8)] for k, v in self.action_type.items()], key=lambda x: x[1])
        val = list(zip(*action_type))[1]
        action_type.insert(0, ['total', sum(val) / (len(val) + 1e-8)])
        hard_case_action_type = sorted([[k, v] for k, v in self.action_type_hard_case.items()],
                                       key=lambda x: x[1], reverse=True)
        return {
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

    def to_string(self, data=None):
        if data is None:
            data = self.get_stat()
        s = "\n"
        for k, v in data.items():
            if 'hard case' in k:
                v_str = ['{}({})'.format(ACTIONS_REORDER_INV[name], count) for name, count in v]
                s += '{}: {}\n'.format(k, '\t'.join(v_str))
            else:
                v_str = ['{}({:.4f})'.format(name, val) for name, val in v]
                s += '{}: {}\n'.format(k, '\t'.join(v_str))
        return s


class AlphastarSLLearner(SLLearner):
    '''
        Overview: Alphastar implementation inherits from SLLearner, including basic processes.
        Interface: __init__
    '''

    def __init__(self, *args, **kwargs):
        '''
            Overview: initialization method, using setting to build model, dataset, optimizer, lr_scheduler
                      and other helper. It can alse load checkpoint.
        '''
        self.loss_func = {  # multi loss to be calculate
            'action_type': self._criterion_apply,
            'delay': self._delay_loss,
            'queued': self._queued_loss,
            'selected_units': self._selected_units_loss,
            'target_units': self._target_units_loss,
            'target_location': self._target_location_loss,
        }  # must execute before super __init__
        self.data_stat = {
            'action_type': [k for k in ACTIONS],
            'delay': ['0-5', '6-22', '23-44', '44-64'],
            'queued': ['no_attr', 'no_queued', 'queued'],
            'selected_units': ['no_attr', '1', '2-8', '9-32', '33-64', '64+'],
            'target_units': ['no_attr', 'target_units'],
            'target_location': ['no_attr', 'target_location'],
        }  # must execute before super __init__
        super(AlphastarSLLearner, self).__init__(*args, **kwargs)
        self.temperature_scheduler = build_temperature_scheduler(self.cfg)  # get naive temperature scheduler
        self._get_loss = self.time_helper.wrapper(self._get_loss)  # use time helper to calculate forward time
        self.use_value_network = 'value' in self.cfg.model.keys()  # if value in self.cfg.model.keys(), use_value_network=True  # noqa
        self.criterion = build_criterion(self.cfg.train.criterion)  # define loss function
        self.location_expand_ratio = self.cfg.model.policy.location_expand_ratio
        self.eval_criterion = AlphastarSLCriterion()

    # overwrite
    def _init(self):
        '''
            Overview: initialize logger
        '''
        super()._init()
        self.variable_record.register_var('total_loss')
        self.tb_logger.register_var('total_loss')
        for k in self.loss_func.keys():
            self.variable_record.register_var(k + '_loss')
            self.tb_logger.register_var(k + '_loss')

        self.variable_record.register_var('action_type', var_type='1darray', var_item_keys=self.data_stat['action_type'])  # noqa
        self.tb_logger.register_var('action_type', var_type='histogram')
        for k in (set(self.data_stat.keys()) - {'action_type'}):
            self.variable_record.register_var(k, var_type='1darray', var_item_keys=self.data_stat[k])
            self.tb_logger.register_var(k, var_type='scalars')

    # overwrite
    def _get_loss(self, data):
        '''
            Overview: Overwrite function, main process of training
            Arguments:
                - data (:obj:`batch_data`): batch_data created by dataloader
        '''
        if self.use_value_network:  # value network to be implemented
            raise NotImplementedError

        temperature = self.temperature_scheduler.step()
        # reset prev_state
        prev_state = []
        B = len(data[0]['start_step'])
        if not hasattr(self, 'prev_state'):  # init self.prev_state
            self.prev_state = [None for _ in range(B)]
        for prev, is_start in zip(self.prev_state, data[0]['start_step']):
            if is_start:
                prev_state.append(None)
            else:
                prev_state.append(prev)
        prev_state_id = [i for i in range(B)]
        # forward
        # Note: the 1st dim of prev_state is batch size
        loss_items = {k + '_loss': [] for k in self.loss_func.keys()}  # loss name + '_loss'
        for i, step_data in enumerate(data):
            end_index = sorted(step_data['end_index'], reverse=True)
            for i in end_index:
                real_i = prev_state_id.index(i)
                prev_state.pop(real_i)
            prev_state_id = [t for t in prev_state_id if t not in end_index]
            actions = step_data['actions']
            step_data['prev_state'] = prev_state
            policy_logits, prev_state = self.model(step_data, mode='mimic', temperature=temperature)

            for k in loss_items.keys():
                kp = k[:-5]
                loss_items[k].append(self.loss_func[kp](policy_logits[kp], actions[kp]))  # calculate loss
        # record prev_state
        self._update_prev_state(prev_state, end_index)

        for k, v in loss_items.items():
            loss_items[k] = sum(v) / (1e-9 + len(v))
            if not isinstance(loss_items[k], torch.Tensor):
                loss_items[k] = torch.tensor([loss_items[k]], dtype=self.dtype, device=self.device)
        loss_items['total_loss'] = sum(loss_items.values())
        return loss_items

    def _update_prev_state(self, prev_state, end_index):
        end_index = sorted(end_index)
        for idx in end_index:
            prev_state.insert(idx, None)
        next_state = []
        for prev in prev_state:
            if prev is None:
                next_state.append(None)
            else:
                next_state.append([prev[0].detach(), prev[1].detach()])
        self.prev_state = next_state

    # overwrite
    def _get_data_stat(self, data):
        data_stat = {k: {t: 0 for t in v} for k, v in self.data_stat.items()}
        for step_data in data:
            action = step_data['actions']
            for k, v in action.items():
                if k == 'action_type':
                    for t in v:
                        data_stat[k][ACTIONS_REORDER_INV[t.item()]] += 1
                elif k == 'delay':
                    for t in v:
                        if t <= 5:
                            data_stat[k]['0-5'] += 1
                        elif t <= 22:
                            data_stat[k]['6-22'] += 1
                        elif t <= 44:
                            data_stat[k]['23-44'] += 1
                        elif t <= 64:
                            data_stat[k]['44-64'] += 1
                        else:
                            raise ValueError("invalid delay value: {}".format(t))
                elif k == 'queued':
                    for t in v:
                        if not isinstance(t, torch.Tensor):
                            data_stat[k]['no_attr'] += 1
                        elif t == 0:
                            data_stat[k]['no_queued'] += 1
                        elif t == 1:
                            data_stat[k]['queued'] += 1
                        else:
                            raise ValueError("invalid queued value: {}".format(t))
                elif k == 'selected_units':
                    for t in v:
                        if not isinstance(t, torch.Tensor):
                            data_stat[k]['no_attr'] += 1
                        else:
                            num = t.shape[0]
                            if num <= 0:
                                raise ValueError("invalid queued value: {}".format(t))
                            elif num <= 1:
                                data_stat[k]['1'] += 1
                            elif num <= 8:
                                data_stat[k]['2-8'] += 1
                            elif num <= 32:
                                data_stat[k]['9-32'] += 1
                            elif num <= 64:
                                data_stat[k]['33-64'] += 1
                            else:
                                data_stat[k]['64+'] += 1
                elif k == 'target_units':
                    for t in v:
                        if not isinstance(t, torch.Tensor):
                            data_stat[k]['no_attr'] += 1
                        else:
                            data_stat[k]['target_units'] += 1
                elif k == 'target_location':
                    for t in v:
                        if not isinstance(t, torch.Tensor):
                            data_stat[k]['no_attr'] += 1
                        else:
                            data_stat[k]['target_location'] += 1
        data_stat = {k: list(v.values()) for k, v in data_stat.items()}
        return data_stat

    # overwrite
    def _record_additional_info(self, iterations):
        histogram_keys = ['action_type']
        scalars_keys = self.data_stat.keys() - histogram_keys
        self.tb_logger.add_val_list(self.variable_record.get_vars_tb_format(
            scalars_keys, iterations, var_type='1darray', viz_type='scalars'), viz_type='scalars')
        self.tb_logger.add_val_list(self.variable_record.get_vars_tb_format(
            histogram_keys, iterations, var_type='1darray', viz_type='histogram'), viz_type='histogram')

    def _criterion_apply(self, logits, labels):
        '''
            Overview: calculate CrossEntropyLoss of taking each action or each delay
            Arguments:
                - logits (:obj:`tensor`): The logits corresponding to the probabilities of
                                          taking each action or each delay
                - labels (:obj:`list`): label from batch_data, list[Tensor](len=batch size)
            Returns:
                - (:obj`tensor`): criterion result
        '''
        if isinstance(labels, collections.Sequence):
            labels = torch.cat(labels, dim=0)
        self.device = logits.device
        self.dtype = logits.dtype
        return self.criterion(logits, labels)

    def _delay_loss(self, preds, labels):
        '''
            Overview: calculate L1 loss of taking each action or each delay
            Arguments:
                - preds (:obj:`tensor`): the predict delay
                - labels (:obj:`list`): label from batch_data, list[Tensor](len=batch size)
            Returns:
                - (:obj`tensor`): delay loss result
        '''
        def delay_l1(p, l):
            base = -1.73e-5*l**3 + 1.89e-3*l**2 - 5.8e-2*l + 0.61
            loss = torch.abs(p - l) - base*l
            return loss.clamp(0).mean()
        if isinstance(labels, collections.Sequence):
            labels = torch.cat(labels, dim=0)
        labels = labels.to(preds.dtype)
        assert(preds.shape == labels.shape)
        return delay_l1(preds, labels)

    def _queued_loss(self, logits, labels):
        '''
            Overview: calculate CrossEntropyLoss of queueing
            Arguments:
                - logits (:obj:`tensor`): The logits corresponding to the probabilities of
                                          queued and not queued
                - labels (:obj:`list`): label from batch_data, list[Tensor](len=batch size)
            Returns:
                - (:obj`tensor`): criterion result
        '''
        labels = [x for x in labels if isinstance(x, torch.Tensor)]
        if len(labels) == 0:
            return 0
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)
        return self.criterion(logits, labels)

    def _selected_units_loss(self, logits, labels):
        '''
            Overview: use CrossEntropyLoss between logits and labels
            Arguments:
                - logits (:obj:`tensor`): The logits corresponding to the probabilities of selecting
                                          each unit, repeated for each of the possible 64 unit selections
                - labels (:obj:`list`): label from batch_data, list[Tensor](len=batch size)
            Returns:
                - (:obj`tensor`): criterion result
        '''
        criterion = MultiLogitsLoss(self.cfg.train.criterion)
        labels = [x for x in labels if isinstance(x, torch.Tensor)]
        if len(labels) == 0:
            return 0
        loss = []
        for b in range(len(labels)):
            lo, la = logits[b], labels[b]
            lo = torch.cat(lo, dim=0)
            if lo.shape[0] != la.shape[0]:
                assert(lo.shape[0] == 1 + la.shape[0])  # ISSUE(zm) why?
                end_flag_label = torch.LongTensor([lo.shape[1]-1]).to(la.device)
                end_flag_loss = self.criterion(lo[-1:], end_flag_label)
                logits_loss = criterion(lo[:-1], la)
                loss.append((end_flag_loss + logits_loss) / 2)
            else:
                loss.append(criterion(lo, la))
        return sum(loss) / len(loss)

    def _target_units_loss(self, logits, labels):
        '''
            Overview: calculate CrossEntropyLoss of targeting a unit
            Arguments:
                - logits (:obj:`tensor`): The logits corresponding to the probabilities of targeting a unit
                - labels (:obj:`list`): label from batch_data, list[Tensor](len=batch size)
            Returns:
                - (:obj`tensor`): criterion result
        '''
        labels = [x for x in labels if isinstance(x, torch.Tensor)]
        if len(labels) == 0:
            return 0
        loss = []
        for b in range(len(labels)):
            lo, la = logits[b], labels[b]
            loss.append(self.criterion(lo, la))
        return sum(loss) / len(loss)

    def _target_location_loss(self, logits, labels):
        '''
            Overview: get logits based on resolution and calculate CrossEntropyLoss between logits and label.
            Arguments:
                - logits (:obj:`tensor`): The logits corresponding to the probabilities of targeting each location
                - labels (:obj:`list`): label from batch_data, list[Tensor](len=batch size)
            Returns:
                - (:obj`tensor`): criterion result
        '''
        labels = [x for x in labels if isinstance(x, torch.Tensor)]
        if len(labels) == 0:
            return 0
        ratio = self.location_expand_ratio
        loss = []
        for logit, label in zip(logits, labels):
            logit = F.avg_pool2d(logit, kernel_size=ratio, stride=ratio)  # achieve same resolution with label
            logit.mul_(ratio*ratio)
            H, W = logit.shape[2:]
            label = torch.LongTensor([label[0]*W+label[1]]).to(device=logit.device)  # (y, x)
            logit = logit.view(1, -1)
            loss.append(self.criterion(logit, label))
        return sum(loss) / len(loss)

    # overwrite
    def eval(self):
        self.model.eval()
        for idx, data in enumerate(self.eval_dataloader):
            next_state = None
            for s_idx, step_data in enumerate(data):
                if self.use_cuda:
                    step_data = to_device(step_data, 'cuda')
                step_data['prev_state'] = next_state
                with torch.no_grad():
                    ret = self.model(step_data, mode='evaluate')
                actions, next_state = ret['actions'], ret['next_state']
                self.eval_criterion.update(actions, step_data['actions'])
                if s_idx % 100 == 0:
                    args = [self.rank, idx+1, len(self.eval_dataloader), s_idx, len(data)]
                    self.logger.info('EVAL[rank: {}](sample: {}/{})(step: {}/{})'.format(*args))
        eval_result = self.eval_criterion.get_stat()
        self.logger.info(self.eval_criterion.to_string(eval_result))
        self.model.train()
