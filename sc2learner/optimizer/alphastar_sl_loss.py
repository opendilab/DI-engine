"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. base class for supervised learning on linklink, including basic processes.
"""
import collections

import torch
import torch.nn.functional as F

from sc2learner.optimizer.base_loss import BaseLoss
from sc2learner.torch_utils import MultiLogitsLoss, build_criterion
from pysc2.lib.static_data import BASE_ACTIONS, PART_ACTIONS_MAP


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


class AlphaStarSupervisedLoss(BaseLoss):
    def __init__(self, agent, train_config, model_config):
        # multiple loss to be calculate
        self.loss_func = {
            'action_type': self._action_type_loss,
            'delay': self._delay_loss,
            'queued': self._queued_loss,
            'selected_units': self._selected_units_loss,
            'target_units': self._target_units_loss,
            'target_location': self._target_location_loss,
        }

        self.agent = agent

        self.use_value_network = model_config.use_value_network
        self.location_output_type = model_config.policy.head.location_head.output_type

        self.criterion_config = train_config.criterion
        self.criterion = build_criterion(train_config.criterion)
        self.temperature_scheduler = build_temperature_scheduler(
            train_config.temperature
        )  # get naive temperature scheduler

    def register_log(self, variable_record, tb_logger):
        for k in self.loss_func.keys():
            variable_record.register_var(k + '_loss')
            tb_logger.register_var(k + '_loss')

    def compute_loss(self, data):
        """
            Overview: Overwrite function, main process of training
            Arguments:
                - data (:obj:`batch_data`): batch_data created by dataloader

            FIXME(pzh): I don't think we need to compute loss inside the loss class. Instead, we should compute loss
             in optimizer. The loss should only provide a function to operate on a set of existing logits.
        """
        if self.use_value_network:  # value network to be implemented
            raise NotImplementedError
        self.device = data[0]['spatial_info'].device
        self.dtype = data[0]['spatial_info'].dtype

        temperature = self.temperature_scheduler.step()
        self.agent.reset_previous_state(data[0]["start_step"])

        loss_dict = collections.defaultdict(list)

        for i, step_data in enumerate(data):
            _, policy_logits, _ = self.agent.compute_action(step_data, mode='mimic', temperature=temperature)
            for loss_item_name, loss_func in self.loss_func.items():
                loss_name = "{}_loss".format(loss_item_name)
                loss_dict[loss_name].append(
                    loss_func(policy_logits[loss_item_name], step_data["actions"][loss_item_name])
                )

        new_loss_dict = dict()
        for loss_name, loss_val_list in loss_dict.items():
            assert loss_val_list
            loss_val = sum(loss_val_list) / len(loss_val_list)

            if isinstance(loss_val, torch.Tensor):
                new_loss_dict[loss_name] = loss_val
            else:
                new_loss_dict[loss_name] = torch.tensor([loss_val], dtype=self.dtype, device=self.device)

        new_loss_dict['total_loss'] = sum(new_loss_dict.values())
        return new_loss_dict

    def _action_type_loss(self, logits, labels):
        '''
            Overview: calculate CrossEntropyLoss of taking each action_type
            Arguments:
                - logits (:obj:`tensor`): The logits corresponding to the probabilities of
                                          taking each action_type
                - labels (:obj:`list`): label from batch_data, list[Tensor](len=batch size)
            Returns:
                - (:obj`tensor`): criterion result
        '''
        loss_type = self.criterion_config.action_type
        assert (loss_type in ['normal', 'double_head'])
        if isinstance(labels, collections.Sequence):
            labels = torch.cat(labels, dim=0)
        if loss_type == 'normal':
            return self.criterion(logits, labels)
        elif loss_type == 'double_head':
            loss = 0.
            zero_idx, base_idx, spec_idx = [], [], []
            for idx, val in enumerate(labels):
                val = val.item()
                if val == 0:
                    zero_idx.append(idx)
                elif val in BASE_ACTIONS:
                    base_idx.append(idx)
                else:
                    spec_idx.append(idx)

            def fn(x):
                return torch.LongTensor(x).to(self.device)

            zero_idx, base_idx, spec_idx = fn(zero_idx), fn(base_idx), fn(spec_idx)
            zero_labels = torch.index_select(labels, 0, zero_idx)
            if zero_labels.shape[0] > 0:
                base = torch.index_select(logits[0], 0, zero_idx)
                spec = torch.index_select(logits[1], 0, zero_idx)
                loss += self.criterion(base, zero_labels) + self.criterion(spec, zero_labels)

            base_labels = torch.index_select(labels, 0, base_idx)
            if base_labels.shape[0] > 0:
                base = torch.index_select(logits[0], 0, base_idx)
                spec = torch.index_select(logits[1], 0, base_idx)
                transform_base_labels = torch.zeros_like(base_labels)
                for idx, val in enumerate(base_labels):
                    transform_base_labels[idx] = PART_ACTIONS_MAP['base'][val.item()]
                loss += self.criterion(base, transform_base_labels) + self.criterion(
                    spec,
                    torch.zeros_like(transform_base_labels).long()
                )  # noqa

            spec_labels = torch.index_select(labels, 0, spec_idx)
            if spec_labels.shape[0] > 0:
                base = torch.index_select(logits[0], 0, spec_idx)
                spec = torch.index_select(logits[1], 0, spec_idx)
                transform_spec_labels = torch.zeros_like(spec_labels)
                for idx, val in enumerate(spec_labels):
                    transform_spec_labels[idx] = PART_ACTIONS_MAP['spec'][val.item()]
                loss += self.criterion(spec, transform_spec_labels) + self.criterion(
                    base,
                    torch.zeros_like(transform_spec_labels).long()
                )  # noqa
            return loss

    def _delay_loss(self, preds, labels):
        """
            Overview: calculate L1 loss of taking each action or each delay
            Arguments:
                - preds (:obj:`tensor`): the predict delay
                - labels (:obj:`list`): label from batch_data, list[Tensor](len=batch size)
            Returns:
                - (:obj`tensor`): delay loss result
        """
        def delay_l1(p, l):
            base = -1.73e-5 * l**3 + 1.89e-3 * l**2 - 5.8e-2 * l + 0.61
            loss = torch.abs(p - l) - base * l
            return loss.clamp(0).mean()

        if isinstance(labels, collections.Sequence):
            labels = torch.stack(labels, dim=0)
        labels = labels.to(preds.dtype)
        assert (preds.shape == labels.shape)
        return delay_l1(preds, labels)

    def _queued_loss(self, logits, labels):
        """
            Overview: calculate CrossEntropyLoss of queueing
            Arguments:
                - logits (:obj:`tensor`): The logits corresponding to the probabilities of
                                          queued and not queued
                - labels (:obj:`list`): label from batch_data, list[Tensor](len=batch size)
            Returns:
                - (:obj`tensor`): criterion result
        """
        labels = [x for x in labels if isinstance(x, torch.Tensor)]
        if len(labels) == 0:
            return 0
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)
        return self.criterion(logits, labels)

    def _selected_units_loss(self, logits, labels):
        """
            Overview: use CrossEntropyLoss between logits and labels
            Arguments:
                - logits (:obj:`tensor`): The logits corresponding to the probabilities of selecting
                                          each unit, repeated for each of the possible 64 unit selections
                - labels (:obj:`list`): label from batch_data, list[Tensor](len=batch size)
            Returns:
                - (:obj`tensor`): criterion result
        """
        criterion = MultiLogitsLoss(self.criterion_config)
        labels = [x for x in labels if isinstance(x, torch.Tensor)]
        batch_size = len(labels)
        if batch_size == 0:
            return 0.0

        loss = []
        for batch_index in range(batch_size):
            logit, label = logits[batch_index], labels[batch_index]
            if logit.shape[0] != label.shape[0]:  # when agents selected different number of agents compared to expert
                assert (logit.shape[0] == 1 + label.shape[0])  # ISSUE(zm) why?
                end_flag_label = torch.LongTensor([logit.shape[1] - 1]).to(label.device)
                end_flag_loss = self.criterion(logit[-1:], end_flag_label)
                logits_loss = criterion(logit[:-1], label)
                loss.append((end_flag_loss + logits_loss) / 2)
            else:
                loss.append(criterion(logit, label))
        return sum(loss) / len(loss)

    def _target_units_loss(self, logits, labels):
        """
            Overview: calculate CrossEntropyLoss of targeting a unit
            Arguments:
                - logits (:obj:`tensor`): The logits corresponding to the probabilities of targeting a unit
                - labels (:obj:`list`): label from batch_data, list[Tensor](len=batch size)
            Returns:
                - (:obj`tensor`): criterion result
        """
        labels = [x for x in labels if isinstance(x, torch.Tensor)]
        if len(labels) == 0:
            return 0
        loss = []
        for b in range(len(labels)):
            lo, la = logits[b], labels[b]
            loss.append(self.criterion(lo, la))
        return sum(loss) / len(loss)

    def _target_location_loss(self, logits, labels):
        """
            Overview: get logits based on resolution and calculate CrossEntropyLoss between logits and label.
            Arguments:
                - logits (:obj:`tensor`): The logits corresponding to the probabilities of targeting each location
                - labels (:obj:`list`): label from batch_data, list[Tensor](len=batch size)
            Returns:
                - (:obj`tensor`): criterion result
        """
        labels = [x for x in labels if isinstance(x, torch.Tensor)]
        if len(labels) == 0:
            return 0
        loss = []
        for logit, label in zip(logits, labels):
            if self.location_output_type == 'cls':
                H, W = logit.shape[2:]
                label = torch.LongTensor([label[0] * W + label[1]]).to(device=logit.device)  # (y, x)
                logit = logit.view(1, -1)
                loss.append(self.criterion(logit, label))
            elif self.location_output_type == 'soft_argmax':
                label = label.to(logit.dtype)
                loss.append(F.mse_loss(logit, label))
        return sum(loss) / len(loss)
