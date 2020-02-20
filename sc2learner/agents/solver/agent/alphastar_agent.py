import torch
from .agent import BaseAgent
from sc2learner.agents.model import build_model
from sc2learner.utils import to_device, build_checkpoint_helper


class AlphastarAgent(BaseAgent):

    def __init__(self, cfg):
        self.cfg = cfg
        self.out_res = cfg.model.output_resolution
        self.model = build_model(cfg)
        self.model.eval()
        self.use_cuda = cfg.train.use_cuda
        if self.use_cuda:
            self.model = to_device(self.model, 'cuda')

        self.next_state = None
        self.checkpoint_helper = build_checkpoint_helper(cfg)
        self.checkpoint_helper.load(cfg.common.load_path, self.model, prefix='module.', prefix_op='remove')

    def reset(self):
        self.next_state = None

    def act(self, obs):
        entity_raw, map_size = obs['entity_raw'], obs['map_size']
        obs['prev_state'] = self.next_state
        if self.use_cuda:
            obs = to_device(obs, 'cuda')
        obs = self._unsqueeze_batch_dim(obs)
        with torch.no_grad():
            ret = self.model(obs, mode='evaluate')
        actions, self.next_state = ret['actions'], ret['next_state']
        if self.use_cuda:
            actions = to_device(actions, 'cpu')
        actions = self._decode_action(actions, entity_raw, map_size)
        return actions

    def value(self, obs):
        return 0

    def _unsqueeze_batch_dim(self, obs):
        def unsqueeze(x):
            if isinstance(x, dict):
                for k in x.keys():
                    x[k] = x[k].unsqueeze(0)
            elif isinstance(x, torch.Tensor):
                x = x.unsqueeze(0)
            else:
                raise TypeError("invalid type: {}".format(type(x)))
            return x

        unsqueeze_keys = ['scalar_info', 'spatial_info']
        list_keys = ['entity_info', 'entity_raw']
        for k, v in obs.items():
            if k in unsqueeze_keys:
                obs[k] = unsqueeze(v)
            if k in list_keys:
                obs[k] = [obs[k]]
        return obs

    def _decode_action(self, actions, entity_raw, map_size):
        for k, v in actions.items():
            actions[k] = v[0]  # remove batch size dim(batch size=1)
        for k, v in actions.items():
            if k == 'selected_units' or k == 'target_units':
                if isinstance(v, torch.Tensor):
                    index = torch.nonzero(v)[0].tolist()
                    units = []
                    for i in index:
                        units.append(entity_raw['id'][i])
                    actions[k] = torch.LongTensor(units)
            elif k == 'target_location':
                if isinstance(v, torch.Tensor):
                    v = v.item()
                    location = [v // self.out_res[1], v % self.out_res[1]]
                    location[0] = location[0]*1.0 / self.out_res[0] * map_size[0]
                    location[1] = location[1]*1.0 / self.out_res[1] * map_size[1]
                    actions[k] = torch.LongTensor(location)
        return actions
