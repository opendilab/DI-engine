import math
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sl_learner import SLLearner
from sc2learner.dataset import policy_collate_fn
from sc2learner.nn_utils import MultiLogitsLoss


def build_temperature_scheduler(cfg):
    class NaiveT(object):
        def __init__(self, init_val=0.1):
            self.init_val = init_val

        def step(self):
            return self.init_val

    return NaiveT(init_val=cfg.train.temperature)


class AlphastarSLLearner(SLLearner):
    def __init__(self, *args, **kwargs):
        super(AlphastarSLLearner, self).__init__(*args, **kwargs)
        setattr(self.dataloader, 'collate_fn', policy_collate_fn)
        self.temperature_scheduler = build_temperature_scheduler(self.cfg)
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
    def _get_loss(self, data):
        if self.use_value_network:
            raise NotImplementedError

        temperature = self.temperature_scheduler.step()
        prev_state = None
        loss_func = {
            'action_type': self._criterion_apply,
            'delay': self._criterion_apply,
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
            actions = step_data['actions']
            step_data['prev_state'] = prev_state
            policy_logits, prev_state = self.model(step_data, mode='mimic', temperature=temperature)

            for k in loss_items.keys():
                kp = k[:-5]
                loss_items[k].append(loss_func[kp](policy_logits[kp], actions[kp]))

        for k, v in loss_items.items():
            loss_items[k] = sum(v) / len(v)
            if not isinstance(loss_items[k], torch.Tensor):
                dtype = policy_logits['action_type'].dtype
                device = policy_logits['action_type'].device
                loss_items[k] = torch.tensor([loss_items[k]], dtype=dtype, device=device)
        loss_items['total_loss'] = sum(loss_items.values())
        return loss_items

    def _criterion_apply(self, logits, label):
        if isinstance(label, collections.Sequence):
            label = torch.cat(label, dim=0)
        return self.criterion(logits, label)

    def _queued_loss(self, logits, label):
        label = [x for x in label if isinstance(x, torch.Tensor)]
        if len(label) == 0:
            return 0
        logits = torch.cat(logits, dim=0)
        label = torch.cat(label, dim=0)
        return self.criterion(logits, label)

    def _selected_units_loss(self, logits, label):
        criterion = MultiLogitsLoss(criterion='cross_entropy')
        label = [x for x in label if isinstance(x, torch.Tensor)]
        if len(label) == 0:
            return 0
        loss = []
        for b in range(len(label)):
            lo, la = logits[b], label[b]
            lo = torch.cat(lo, dim=0)
            if lo.shape[0] != la.shape[0]:
                assert(lo.shape[0] == 1 + la.shape[0])
                end_flag_label = torch.LongTensor([lo.shape[1]-1]).to(la.device)
                end_flag_loss = self.criterion(lo[-1:], end_flag_label)
                logits_loss = criterion(lo[:-1], la)
                loss.append((end_flag_loss + logits_loss) / 2)
            else:
                loss.append(criterion(lo, la))
        return sum(loss) / len(loss)

    def _target_units_loss(self, logits, label):
        label = [x for x in label if isinstance(x, torch.Tensor)]
        if len(label) == 0:
            return 0
        loss = []
        for b in range(len(label)):
            lo, la = logits[b], label[b]
            loss.append(self.criterion(lo, la))
        return sum(loss) / len(loss)

    def _target_location_loss(self, logits, label):
        label = [x for x in label if isinstance(x, torch.Tensor)]
        if len(label) == 0:
            return 0
        logits = torch.cat(logits, dim=0)
        label = [x*self.resolution[1]+y for (x, y) in label]
        label = torch.LongTensor(label).to(device=logits.device)
        ratio = math.sqrt(logits.shape[1] / (self.resolution[0]*self.resolution[1]))
        assert(math.fabs(int(ratio) - ratio) < 1e-4)
        ratio = int(ratio)
        B = logits.shape[0]
        N = int(math.sqrt(logits.shape[1]))
        logits = logits.reshape(B, N, N).unsqueeze(1)
        logits = F.avg_pool2d(logits, kernel_size=ratio, stride=ratio)
        logits = logits.squeeze(1).reshape(B, -1)
        return self.criterion(logits, label)
