import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sl_learner import SLLearner
from sc2learner.dataset import policy_collate_fn


class AlphastarSLLearner(SLLearner):
    def __init__(self, *args, **kwargs):
        super(AlphastarSLLearner, self).__init__(*args, **kwargs)
        setattr(self.dataloader, 'collate_fn', policy_collate_fn)
        self._get_loss = self.time_helper.wrapper(self._get_loss)
        self.use_value_network = 'value' in self.cfg.model.keys()
        self.criterion = nn.CrossEntropyLoss()
        self.resolution = self.cfg.data.resolution

    # overwrite
    def _init(self):
        super()._init()
        self.scalar_record.register_var('total_loss')
        self.scalar_record.register_var('action_type_loss')
        self.scalar_record.register_var('delay_loss')
        self.scalar_record.register_var('queued_loss')
        self.scalar_record.register_var('selected_units_loss')
        self.scalar_record.register_var('target_units_loss')
        self.scalar_record.register_var('target_location_loss')

        self.tb_logger.register_var('total_loss')
        self.tb_logger.register_var('action_type_loss')
        self.tb_logger.register_var('delay_loss')
        self.tb_logger.register_var('queued_loss')
        self.tb_logger.register_var('selected_units_loss')
        self.tb_logger.register_var('target_units_loss')
        self.tb_logger.register_var('target_location_loss')

    # overwrite
    def _get_loss(self, data, train_param):
        if self.use_value_network:
            raise NotImplementedError

        prev_state = None
        loss_func = {
            'action_type': self.criterion,
            'delay': self.criterion,
            'queued': self._queued_loss,
            'selected_units': self._selected_units_loss,
            'target_units': self._target_units_loss,
            'target_location': self._target_location_loss,
        }
        loss_items = {
            'action_type_loss': [],
            'delay_loss': [],
            'queued_loss': [],
            'selected_units_loss': [],
            'target_units_loss': [],
            'target_location_loss': []
        }
        for i, step_data in enumerate(data):
            actions = data['actions']
            step_data['prev_state'] = prev_state
            policy_logits, prev_state = self.model(step_data, mode='mimic', temperature=train_param['temperature'])

            for k in loss_items.keys():
                kp = k[:-5]
                loss_items[k].append(loss_func[kp](policy_logits[kp], actions[kp]))

        for k, v in loss_items.items():
            loss_items[k] = sum(v) / len(v)
        loss_items['total_loss'] = sum(loss_items.values())
        return loss_items

    def _queued_loss(self, logits, label):
        label = [x for x in label if x != 'none']
        if len(label) == 0:
            return 0
        logits = torch.cat(logits, dim=0)
        label = torch.cat(label, dim=0)
        return self.criterion(logits, label)

    def _selected_units_loss(self, logits, label):
        label = [x for x in label if x != 'none']
        if len(label) == 0:
            return 0
        loss = []
        for b in range(len(label)):
            lo, la = logits[b], label[b]
            # TODO min CE match
            lo = torch.cat(lo, dim=0)
            la = torch.cat(la, dim=0)
            loss.append(self.criterion(lo, la))
        return sum(loss) / len(loss)

    def _target_units_loss(self, logits, label):
        return self._selected_units_loss(logits, label)

    def _target_location_loss(self, logits, label):
        label = [x for x in label if x != 'none']
        if len(label) == 0:
            return 0
        logits = torch.cat(logits, dim=0)
        label = [x*self.resolution[1]+y for (x, y) in label]
        label = torch.cat(label, dim=0)
        ratio = math.sqrt(self.resolution[0]*self.resolution[1] / logits.shape[1])
        assert(math.fbs(int(ratio) - ratio) < 1e-4)
        ratio = int(ratio)
        logits = F.avg_pool2d(logits, kernel_size=ratio, stride=ratio)
        return self.criterion(logits, label)
