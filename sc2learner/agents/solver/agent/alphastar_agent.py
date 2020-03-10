import torch
from .agent import BaseAgent
from sc2learner.agents.model import build_model
from sc2learner.utils import to_device, build_checkpoint_helper
from pysc2.lib.static_data import ACTIONS_REORDER_INV


class AlphastarAgent(BaseAgent):

    def __init__(self, cfg):
        self.cfg = cfg
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
                    if isinstance(x[k], dict):
                        for kk in x[k].keys():
                            x[k][kk] = x[k][kk].unsqueeze(0)
                    else:
                        x[k] = x[k].unsqueeze(0)
            elif isinstance(x, torch.Tensor):
                x = x.unsqueeze(0)
            else:
                raise TypeError("invalid type: {}".format(type(x)))
            return x

        unsqueeze_keys = ['scalar_info', 'spatial_info']
        list_keys = ['entity_info', 'entity_raw', 'map_size']
        for k, v in obs.items():
            if k in unsqueeze_keys:
                obs[k] = unsqueeze(v)
            if k in list_keys:
                obs[k] = [obs[k]]
        return obs

    def _decode_action(self, actions, entity_raw, map_size):
        for k, v in actions.items():
            val = v[0]  # remove batch size dim(batch size=1)
            if isinstance(val, torch.Tensor):
                if k == 'selected_units' or k == 'target_units':
                    actions[k] = [entity_raw['id'][i] for i in val]
                elif k == 'action_type':
                    actions[k] = ACTIONS_REORDER_INV[val.item()]
                elif k == 'delay' or k == 'queued':
                    actions[k] = val.item()
                elif k == 'target_location':
                    actions[k] = val.tolist()
                else:
                    raise KeyError("invalid key:{}".format(k))
            else:
                actions[k] = val
        return actions
