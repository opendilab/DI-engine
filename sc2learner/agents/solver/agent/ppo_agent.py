import torch
from .agent import BaseAgent
import matplotlib.pyplot as plt
import numpy as np


class PpoAgent(BaseAgent):

    def __init__(self, *args, **kwargs):
        super(PpoAgent, self).__init__(*args, **kwargs)
        self.plt_count = 0
        self.tb_logger.register_var('logits', var_type='figure')

    def reset(self):
        self.state = self.model.initial_state
        self.done = False

    def act(self, obs):
        inputs = self._pack_model_input(obs)
        ret = self.model(inputs, mode="step")
        action = ret['action']
        if self.viz:
            viz_feature = ret['viz_feature']
            keys = viz_feature.keys()
            B, N = viz_feature['logits'].shape
            x = np.linspace(0, N, N)
            figure = plt.figure()
            valid_mask = []
            for b in range(B):
                v = viz_feature['logits'][b]
                valid_mask.append(v.gt(-1000))  # mask out unavailable action
            for b in range(B):
                for k in keys:
                    v = viz_feature[k][b]
                    v = torch.masked_select(v, valid_mask[b])
                    plt.scatter(x, v, alpha=0.6, s=50, label=k)
                    max_v, min_v = v.max().item(), v.min().item()
                    plt.ylim((min_v-1, max_v+1))
                plt.legend(loc='upper right')
                self.tb_logger.add_figure('logits', figure, self.plt_count, close=True)
                self.plt_count += 1
        return action.item()

    def value(self, obs):
        inputs = self._pack_model_input(obs)
        value = self.model(inputs, mode='value')['value']
        return value.item()

    def _pack_model_input(self, obs):
        inputs = {}
        if self.model.use_mask:
            observation, mask = obs
            inputs['mask'] = torch.FloatTensor(mask).unsqueeze(0)
        else:
            observation = obs
        observation = torch.FloatTensor(observation).unsqueeze(0)
        done = torch.FloatTensor([self.done]).unsqueeze(0)
        inputs['obs'] = observation
        inputs['done'] = done
        if self.cfg.model.policy == 'lstm':
            inputs['state'] = self.state.unsqueeze(0)
        return inputs
