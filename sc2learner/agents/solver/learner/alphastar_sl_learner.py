'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. Alphastar implementation for supervised learning on linklink, including basic processes.
'''
import math
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sl_learner import SLLearner
from pysc2.lib.static_data import ACTIONS
from sc2learner.dataset import policy_collate_fn
from sc2learner.nn_utils import MultiLogitsLoss


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
            'delay': self._criterion_apply,
            'queued': self._queued_loss,
            'selected_units': self._selected_units_loss,
            'target_units': self._target_units_loss,
            'target_location': self._target_location_loss,
        }  # must execute before super __init__
        self.data_stat = {
            'action_type': [k for k in ACTIONS],
            'delay': ['0-16', '17-32', '33-64', '65-128'],
            'queued': ['no_attr', 'no_queued', 'queued'],
            'selected_units': ['no_attr', '1', '2-8', '9-32', '33-64', '64+'],
            'target_units': ['no_attr', 'target_units'],
            'target_location': ['no_attr', 'target_location'],
        }  # must execute before super __init__
        super(AlphastarSLLearner, self).__init__(*args, **kwargs)
        setattr(self.dataloader, 'collate_fn', policy_collate_fn)  # use dataloader.collate_fn to call this function
        self.temperature_scheduler = build_temperature_scheduler(self.cfg)  # get naive temperature scheduler
        self._get_loss = self.time_helper.wrapper(self._get_loss)  # use time helper to calculate forward time
        self.use_value_network = 'value' in self.cfg.model.keys()  # if value in self.cfg.model.keys(), use_value_network=True  # noqa
        self.criterion = nn.CrossEntropyLoss()  # define loss function
        self.resolution = self.cfg.data.resolution

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
        prev_state = None  # previous LSTM state from model
        loss_items = {k + '_loss': [] for k in self.loss_func.keys()}  # loss name + '_loss'
        for i, step_data in enumerate(data):
            actions = step_data['actions']
            step_data['prev_state'] = prev_state
            policy_logits, prev_state = self.model(step_data, mode='mimic', temperature=temperature)

            for k in loss_items.keys():
                kp = k[:-5]
                loss_items[k].append(self.loss_func[kp](policy_logits[kp], actions[kp]))  # calculate loss

        for k, v in loss_items.items():
            loss_items[k] = sum(v) / len(v)
            if not isinstance(loss_items[k], torch.Tensor):
                dtype = policy_logits['action_type'].dtype
                device = policy_logits['action_type'].device
                loss_items[k] = torch.tensor([loss_items[k]], dtype=dtype, device=device)
        loss_items['total_loss'] = sum(loss_items.values())
        return loss_items

    # overwrite
    def _get_data_stat(self, data):
        data_stat = {k: {t: 0 for t in v} for k, v in self.data_stat.items()}
        for step_data in data:
            action = step_data['actions']
            for k, v in action.items():
                if k == 'action_type':
                    for t in v:
                        data_stat[k][t.item()] += 1
                elif k == 'delay':
                    for t in v:
                        if t <= 16:
                            data_stat[k]['0-16'] += 1
                        elif t <= 32:
                            data_stat[k]['17-32'] += 1
                        elif t <= 64:
                            data_stat[k]['33-64'] += 1
                        elif t <= 128:
                            data_stat[k]['65-128'] += 1
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

    def _criterion_apply(self, logits, label):
        '''
            Overview: calculate CrossEntropyLoss of taking each action or each delay
            Arguments:
                - logits (:obj:`tensor`): The logits corresponding to the probabilities of
                                          taking each action or each delay
                - label (:obj:`tensor`): label from batch_data
            Returns:
                - (:obj`tensor`): criterion result
        '''
        if isinstance(label, collections.Sequence):
            label = torch.cat(label, dim=0)
        return self.criterion(logits, label)

    def _queued_loss(self, logits, label):
        '''
            Overview: calculate CrossEntropyLoss of queueing
            Arguments:
                - logits (:obj:`tensor`): The logits corresponding to the probabilities of
                                          queueing and not queueing
                - label (:obj:`tensor`): label from batch_data
            Returns:
                - (:obj`tensor`): criterion result
        '''
        label = [x for x in label if isinstance(x, torch.Tensor)]
        if len(label) == 0:
            return 0
        logits = torch.cat(logits, dim=0)
        label = torch.cat(label, dim=0)
        return self.criterion(logits, label)

    def _selected_units_loss(self, logits, label):
        '''
            Overview: use CrossEntropyLoss between logits and label
            Arguments:
                - logits (:obj:`tensor`): The logits corresponding to the probabilities of selecting
                                          each unit, repeated for each of the possible 64 unit selections
                - label (:obj:`tensor`): label from batch_data
            Returns:
                - (:obj`tensor`): criterion result
        '''
        criterion = MultiLogitsLoss(criterion='cross_entropy')  #
        label = [x for x in label if isinstance(x, torch.Tensor)]
        if len(label) == 0:
            return 0
        loss = []
        for b in range(len(label)):
            lo, la = logits[b], label[b]
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

    def _target_units_loss(self, logits, label):
        '''
            Overview: calculate CrossEntropyLoss of targeting a unit
            Arguments:
                - logits (:obj:`tensor`): The logits corresponding to the probabilities of targeting a unit
                - label (:obj:`tensor`): label from batch_data
            Returns:
                - (:obj`tensor`): criterion result
        '''
        label = [x for x in label if isinstance(x, torch.Tensor)]
        if len(label) == 0:
            return 0
        loss = []
        for b in range(len(label)):
            lo, la = logits[b], label[b]
            loss.append(self.criterion(lo, la))
        return sum(loss) / len(loss)

    def _target_location_loss(self, logits, label):
        '''
            Overview: get logits based on resolution and calculate CrossEntropyLoss between logits and label.
            Arguments:
                - logits (:obj:`tensor`): The logits corresponding to the probabilities of targeting each location
                - label (:obj:`tensor`): label from batch_data
            Returns:
                - (:obj`tensor`): criterion result
        '''
        label = [x for x in label if isinstance(x, torch.Tensor)]
        if len(label) == 0:
            return 0
        logits = torch.cat(logits, dim=0)  # probabilities of targeting each location
        label = [x*self.resolution[1]+y for (x, y) in label]  # location of targeting
        label = torch.LongTensor(label).to(device=logits.device)
        ratio = math.sqrt(logits.shape[1] / (self.resolution[0]*self.resolution[1]))  # must be integer
        assert(math.fabs(int(ratio) - ratio) < 1e-4)
        ratio = int(ratio)
        B = logits.shape[0]
        N = int(math.sqrt(logits.shape[1]))
        logits = logits.reshape(B, N, N).unsqueeze(1)
        logits = F.avg_pool2d(logits, kernel_size=ratio, stride=ratio)  # achieve same resolution with label
        logits = logits.squeeze(1).reshape(B, -1)
        return self.criterion(logits, label)
