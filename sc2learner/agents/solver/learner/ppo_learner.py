import torch
import torch.nn.functional as F
from sc2learner.dataset import OnlineDataset, OnlineDataLoader, unroll_split_collate_fn
from sc2learner.utils import build_logger, build_checkpoint_helper, build_time_helper, to_device
from .learner import BaseLearner


def build_clip_range_scheduler(cfg):
    class NaiveClip(object):
        def __init__(self, init_val=0.1):
            self.init_val = init_val

        def step(self):
            return self.init_val

    return NaiveClip(init_val=cfg.train.ppo_clip_range)


class PpoLearner(BaseLearner):
    def __init__(self, *args, **kwargs):
        super(PpoLearner, self).__init__(*args, **kwargs)
        self.use_value_clip = self.cfg.train.use_value_clip
        self.unroll_split = self.cfg.train.unroll_split if self.model.initial_state is None else 1
        self.entropy_coeff = self.cfg.train.entropy_coeff
        self.value_coeff = self.cfg.train.value_coeff
        self.clip_range_scheduler = build_clip_range_scheduler(self.cfg)
        self.dataloader = OnlineDataLoader(self.dataset, self.cfg.train.batch_size,
                                           collate_fn=unroll_split_collate_fn)
        self.enable_save_data = self.cfg.train.enable_save_data
        self.data_count = 0  # TODO add mutex to synchronize count behaviour(multi thread pull data)
        if self.cfg.common.data_load_path != '':
            self.dataset.load_data(self.cfg.common.data_load_path, ratio=self.unroll_split)  # 560 data 20 second
        self._get_loss = self.time_helper.wrapper(self._get_loss)

    # overwrite
    def _init(self):
        super()._init()
        self.variable_record.register_var('total_loss')
        self.variable_record.register_var('pg_loss')
        self.variable_record.register_var('value_loss')
        self.variable_record.register_var('entropy_reg')
        self.variable_record.register_var('approximate_kl')
        self.variable_record.register_var('clipfrac')

        self.tb_logger.register_var('total_loss')
        self.tb_logger.register_var('pg_loss')
        self.tb_logger.register_var('value_loss')
        self.tb_logger.register_var('entropy_reg')
        self.tb_logger.register_var('approximate_kl')
        self.tb_logger.register_var('clipfrac')

    # overwrite
    def _parse_pull_data(self, data):
        # TODO (speed and memory copy optimization)
        keys = ['obs', 'return', 'done', 'action', 'value', 'neglogp', 'state', 'episode_infos', 'model_index']
        if self.model.use_mask:
            keys.append('mask')
        temp_dict = {}
        for k, v in data.items():
            if k in keys:
                if k == 'state':
                    if v is None:
                        temp_dict[k] = [None for _ in range(self.unroll_split)]
                    else:
                        raise NotImplementedError
                elif k == 'episode_infos' or k == 'model_index':
                    temp_dict[k] = [v for _ in range(self.unroll_split)]
                else:
                    stack_item = torch.stack(v, dim=0)
                    split_item = torch.chunk(stack_item, self.unroll_split)
                    temp_dict[k] = split_item
        for i in range(self.unroll_split):
            item = {k: v[i] for k, v in temp_dict.items()}
            self.dataset.push_data(item)
            if self.enable_save_data:
                self.checkpoint_helper.save_data('split_item_{}'.format(self.data_count), item, device='cpu')
                self.data_count += 1

    # overwrite
    def _get_loss(self, data):
        clip_range = self.clip_range_scheduler.step()
        obs, actions, returns, neglogps, values, dones, states = (
            data['obs'], data['action'], data['return'], data['neglogp'],
            data['value'], data['done'], data['state']
        )

        inputs = {}
        inputs['obs'] = obs
        inputs['done'] = dones
        inputs['state'] = states
        inputs['action'] = actions
        if self.model.use_mask:
            inputs['mask'] = data['mask']
        outputs = self.model(inputs, mode='evaluate')
        new_values, new_neglogp, entropy = (
            outputs['value'], outputs['neglogp'], outputs['entropy']
        )

        new_values_clipped = values + torch.clamp(new_values - values, -clip_range, clip_range)
        value_loss1 = torch.pow(new_values - returns, 2)
        if self.use_value_clip:
            value_loss2 = torch.pow(new_values_clipped - returns, 2)
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        else:
            value_loss = 0.5 * value_loss1.mean()

        adv = returns - values
        adv = adv.squeeze(1)  # bug point
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ratio = torch.exp(neglogps - new_neglogp)
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        approximate_kl = 0.5 * torch.pow(new_neglogp - neglogps, 2).mean()
        clipfrac = torch.abs(ratio - 1.0).gt(clip_range).float().mean()

        loss = pg_loss - entropy * self.entropy_coeff + value_loss * self.value_coeff

        loss_items = {}
        loss_items['total_loss'] = loss
        loss_items['approximate_kl'] = approximate_kl
        loss_items['clipfrac'] = clipfrac
        loss_items['pg_loss'] = pg_loss
        loss_items['entropy_reg'] = entropy * self.entropy_coeff
        loss_items['value_loss'] = value_loss * self.value_coeff
        return loss_items

    # overwrite
    def _update_monitor_var(self, items, time_items):
        keys = self.variable_record.get_var_names()
        new_dict = {}
        for k in keys:
            if k in items.keys():
                v = items[k]
                if isinstance(v, torch.Tensor):
                    v = v.item()
                else:
                    v = v
                new_dict[k] = v
        self.variable_record.update_var(new_dict)
        self.variable_record.update_var(time_items)

    # overwrite
    def _data_transform(self, data):
        keys = ['obs', 'return', 'done', 'action', 'value', 'neglogp', 'state']
        if self.model.use_mask:
            keys.append('mask')
        transformed_data = {}
        for k, v in data.items():
            if k in keys:
                if v is None:
                    v = 'none'
                transformed_data[k] = v
        return transformed_data
