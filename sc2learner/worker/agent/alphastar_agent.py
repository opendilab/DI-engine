import torch
from .agent import BaseAgent
from sc2learner.agent.model import build_model
from sc2learner.torch_utils import to_device, build_checkpoint_helper
from sc2learner.utils import dict_list2list_dict
from sc2learner.envs import action_unit_id_transform


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
        entity_raw = obs['entity_raw']
        # preprocessing: merge prev_state->batch->cuda
        obs['prev_state'] = self.next_state
        obs = self._unsqueeze_batch_dim(obs)
        if self.use_cuda:
            obs = to_device(obs, 'cuda')
        # forward
        with torch.no_grad():
            actions, self.next_state = self.model(obs, mode='evaluate')
        # postprocessing: cpu->remove batch->action_unit_id_transform inv
        if self.use_cuda:
            actions = to_device(actions, 'cpu')
        actions = dict_list2list_dict(actions)[0]
        actions = action_unit_id_transform({'actions': actions, 'entity_raw': entity_raw}, inverse=True)['actions']
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
