import torch
from sc2learner.utils import build_checkpoint_helper
from .agent import BaseAgent


class PpoAgent(BaseAgent):

    def __init__(self, *args, **kwargs):
        super(PpoAgent, self).__init__(*args, **kwargs)

    def reset(self):
        self.state = self.model.initial_state
        self.done = False

    def act(self, obs):
        inputs = self._pack_model_input(obs)
        action = self.model(inputs, mode='step')[0]
        return action.squeeze(0).numpy()

    def value(self, obs):
        inputs = self._pack_model_input(obs)
        value = self.model(inputs, mode='value')
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
